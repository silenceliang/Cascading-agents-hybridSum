""" RL training utilities"""
import math
from time import time
from datetime import timedelta
import random
from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.distributions import Bernoulli


from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch, ext_batch in loader:
            ext_sents = []
            ext_inds = []
            sent_acts = []
            for raw_arts in art_batch:
                (indices,_), actions = agent(raw_arts)        
                ext_inds += [(len(ext_sents), len(indices)-1)]
                ext_sents += [raw_arts[idx.item()]
                              for idx in indices if idx.item() < len(raw_arts)]

                sent_acts += [actions[j]
                            for j,idx in enumerate(indices) if idx.item() < len(raw_arts)] 

            assert len(ext_sents) == len(sent_acts)

            all_summs = []
            need_abs_sents = [ext_sents[iters] for iters, act in enumerate(sent_acts) if act==0]
            if len(need_abs_sents) > 0:
                turn_abs_sents = abstractor(need_abs_sents)

            for nums, action in enumerate(sent_acts):
                if action == 0:
                    all_summs += turn_abs_sents.pop(0)
                else:
                    all_summs += ext_sents[nums]

            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, target_agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0):
    opt.zero_grad()
    def length_penalty(l, eps=0.7):
        return  (5 + len(l)-1)**eps / (5+1)**eps

    indices, probs,baselines  = [],[],[]
    target_indices, target_probs,target_baselines  = [],[],[]
    act, act_probs, act_baselines = [],[],[]
    pure_act = []
    ext_sents = [] # 所有句子
    abstract = []
    pres_or_rewr = []
    act_precision = []
    target_probs = []

    true_indices, false_indices = [],[]
    recall, precision = [],[]
    art_batch, abs_batch, ext_indices = next(loader)

    for n,(raw_arts, raw_abs, ext_index) in enumerate(zip(art_batch, abs_batch, ext_indices)):
       
        with torch.no_grad():
            (target_inds, target_ms), target_bs, (target_act_inds, target_act_ms), target_act_bs = target_agent(raw_arts)

        (inds, ms), bs, (act_inds, act_ms), act_bs = agent(raw_arts, observations=target_inds)

        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)

        target_probs.append(target_ms)

        act.append(act_inds)
        act_probs.append(act_ms)
        act_baselines.append(act_bs)
        pure_act += [a.detach() for a in act_inds[:-1]]
        
        def precision_recall():
            """ sentence-level precision/recall """
            true_positive_extracts = []
            false_extracts = []
            # acc_pre = []
            for k, index in enumerate(inds[:-1]):
                if index.item() in ext_index:
                    true_positive_extracts.append(index.detach().item())
                else:
                    false_extracts.append(index.detach().item())
            
            each_recall = (len(true_positive_extracts)) / (len(ext_index))
            each_precision = (len(true_positive_extracts)) / (len(inds)-1) if len(inds)>1 else 0
            recall.append(each_recall)
            precision.append(each_precision)
            true_indices.append(true_positive_extracts)
            false_indices.append(false_extracts)
        precision_recall()
        
        """ 存取原句 """
        current_sentence = [raw_arts[idx.item()]
                      for idx in inds if idx.item() < len(raw_arts)]
        ext_sents +=  current_sentence

    assert len(indices) == len(act)
    assert len(probs) == len(act_probs)
    assert len(baselines) == len(act_baselines)

    """ 呼叫改寫模型 """
    with torch.no_grad():
        rewrite_sentences = abstractor(ext_sents)

    """ 經過abs agent後,重組句子 """
    results = [rewrite_sentences[n] if bins.item()==0 
                else ext_sents[n] for n, bins in enumerate(pure_act)]

    """ 原本選句子該得到的 reward 計算  """
    i = 0
    rewards = []
    avg_reward = 0
    avg_reward_abs_rew = 0
    avg_reward_abs_pre = 0
    rewrite_rewards, preserve_rewards = [], []
    

    for inds, act_inds, abss, labeled in zip(indices, act, abs_batch, ext_indices): 
        """ 全部都做 總得分 """
        """ 原先extractor agent取的原始句子 """
        rs = ([reward_fn(ext_sents[i+j], abss[j]) for j in range(min(len(inds)-1, len(abss)))]
              + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
              + [stop_coeff*stop_reward_fn(
                  list(concat(ext_sents[i:i+len(inds)-1])),
                  list(concat(abss)))])               
        # rs = ([reward_fn(sum(ext_sents[i:i+j+1],[]), sum(abss[:j+1],[])) for j in range(min(len(inds)-1, len(abss)))]
        #       + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
        #       + [stop_coeff*stop_reward_fn(
        #           list(concat(ext_sents[i:i+len(inds)-1])),
        #           list(concat(abss)))])

        """ 單句 """
        def single_compared_MC_TD(choice):   
            
            # before_rewrite = ([stop_reward_fn(ext_sents[i+j], abss[j]) 
            #                     for j in range(min(len(inds)-1, len(abss)))]
            #                 + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
            #                 + [0])
            # after_rewrite = ([stop_reward_fn(rewrite_sentences[i+j], abss[j]) 
            #                     for j in range(min(len(inds)-1, len(abss)))]
            #                 + [1 for _ in range(max(0, len(inds)-1-len(abss)))]
            #                 + [0])
            remain_sents = min(len(inds)-1, len(abss))
            if choice=='TD':
                previous_results = [[]] + results[i:i+remain_sents]
                """  abstractor agent 做完 rewrite 後 能得到的分數 """
                rew_rs = ([reward_fn(results[i+j], abss[j]) - reward_fn(previous_results[j], abss[j]) 
                            if act_inds[j].item()==0 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        # + [reward_fn(sum(results[i:i+remain_sents+j+1],[]), sum(abss[:-1],[])) - reward_fn(sum(results[i:i+remain_sents+j],[]), sum(abss[:-1],[])) 
                        #     if act_inds[j].item()==0 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                        + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                        # + [stop_coeff*stop_reward_fn(
                            # list(concat(
                                # [results[x] for x in range(i, i+len(inds)-1)
                                    # if act_inds[x-i].item()==0])),
                            # list(concat(abss)))])
                rew_rs += [sum(rew_rs)]

                """  abstractor agent 做完 preserve 後 能得到的分數 """
                pres_rs = ([reward_fn(results[i+j], abss[j]) - reward_fn(previous_results[j], abss[j])
                            if act_inds[j].item()==1 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        # + [reward_fn(sum(results[i:i+remain_sents+j+1],[]), sum(abss[:-1],[])) - reward_fn(sum(results[i:i+remain_sents+j],[]), sum(abss[:-1],[])) 
                            # if act_inds[j].item()==1 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                        + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                        # + [stop_coeff*stop_reward_fn(
                            # list(concat(
                                # [results[x] for x in range(i, i+len(inds)-1)
                                    # if act_inds[x-i].item()==1])),
                            # list(concat(abss)))])
                pres_rs += [sum(pres_rs)]

            elif choice=='MC':
                """  abstractor agent 做完 rewrite 後 能得到的分數 """
                rew_rs = ([reward_fn(results[i+j], abss[j]) 
                            if act_inds[j].item()==0 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        # + [reward_fn(results[i+remain_sents+j], sum(abss[:-1],[])) 
                            # if act_inds[j].item()==0 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                        + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                        # + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                rew_rs += [sum(rew_rs)]
                """  abstractor agent 做完 preserve 後 能得到的分數 """
                pres_rs = ([reward_fn(results[i+j], abss[j]) 
                            if act_inds[j].item()==1 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        # + [reward_fn(results[i+remain_sents+j], sum(abss[:-1],[])) 
                            # if act_inds[j].item()==1 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                         + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                pres_rs += [sum(pres_rs)]          
            return rew_rs, pres_rs
        rew_rs, pres_rs = single_compared_MC_TD(choice='TD')
        """ 累計  """
        def accumulated_compared_MC_TD(choice):
            remain_sents = min(len(inds)-1, len(abss))
            if choice=='MC':
                """  abstractor agent 做完 rewrite 後 能得到的分數 """
                rew_rs = ([reward_fn(results[i+j], sum(abss[:j+1],[])) 
                            if act_inds[j].item()==0 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        + [reward_fn(results[i+remain_sents+j], sum(abss[:-1],[])) 
                            if act_inds[j].item()==0 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                        # + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                rew_rs += [sum(rew_rs)]
                """  abstractor agent 做完 preserve 後 能得到的分數 """
                pres_rs = ([reward_fn(results[i+j], sum(abss[:j+1],[])) 
                            if act_inds[j].item()==1 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        + [reward_fn(results[i+remain_sents+j], sum(abss[:-1],[])) 
                            if act_inds[j].item()==1 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                        #  + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                pres_rs += [sum(pres_rs)]
            elif choice=='TD':
                """  abstractor agent 做完 rewrite 後 能得到的分數 """
                rew_rs = ([reward_fn(sum(results[i:i+j+1],[]), sum(abss[:j+1],[])) - reward_fn(sum(results[i:i+j],[]), sum(abss[:j],[])) 
                            if act_inds[j].item()==0 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        + [reward_fn(sum(results[i:i+remain_sents+j+1],[]), sum(abss[:-1],[])) - reward_fn(sum(results[i:i+remain_sents+j],[]), sum(abss[:-1],[])) 
                            if act_inds[j].item()==0 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                        # + [0 for _ in range(max(0, len(inds)-1-len(abss)))])
                rew_rs += [sum(rew_rs)]
                """  abstractor agent 做完 preserve 後 能得到的分數 """
                pres_rs = ([reward_fn(sum(results[i:i+j+1],[]), sum(abss[:j+1],[])) - reward_fn(sum(results[i:i+j],[]), sum(abss[:j],[])) 
                            if act_inds[j].item()==1 else 0 for j in range(min(len(inds)-1, len(abss)))]
                        + [reward_fn(sum(results[i:i+remain_sents+j+1],[]), sum(abss[:-1],[])) - reward_fn(sum(results[i:i+remain_sents+j],[]), sum(abss[:-1],[])) 
                            if act_inds[j].item()==1 else 0 for j in range(max(0, len(inds)-1-len(abss)))])
                pres_rs += [sum(pres_rs)]
            return rew_rs, pres_rs
        # rew_rs, pres_rs = accumulated_compared_MC_TD(choice='MC')

        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        avg_reward_abs_rew += rew_rs[-1]/stop_coeff
        avg_reward_abs_pre += pres_rs[-1]/stop_coeff
        i += len(inds)-1
        
        # compute discounted rewards
        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
        rewrite_rewards += rew_rs
        preserve_rewards += pres_rs

    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    
    target_probs = list(concat(target_probs))

    act = list(concat(act))
    act_probs = list(concat(act_probs))
    act_baselines = list(concat(act_baselines))

    # act_precision = list(concat(act_precision))
    # act_precision = torch.Tensor(act_precision).to(act_baselines[0].device)

    """ 三個隱動作 reward 計算 """
    rewrite_rewards = torch.Tensor(rewrite_rewards).to(act_baselines[0].device)
    rewrite_rewards = (rewrite_rewards - rewrite_rewards.mean()) / (
        rewrite_rewards.std() + float(np.finfo(np.float32).eps))

    preserve_rewards = torch.Tensor(preserve_rewards).to(act_baselines[0].device)
    preserve_rewards = (preserve_rewards - preserve_rewards.mean()) / (
        preserve_rewards.std() + float(np.finfo(np.float32).eps)) 
    
    complex_rewards =  torch.stack([rewrite_rewards, preserve_rewards], dim=-1)

    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (reward.std() + float(np.finfo(np.float32).eps))

    baseline = torch.cat(baselines).squeeze()
    act_baselines = torch.cat(act_baselines).squeeze().view(-1,2)

    assert len(indices)==len(probs)==len(reward)==len(baseline)
    assert len(act)==len(act_probs)==len(act_baselines)
    
    avg_advantage = 0
    losses = []
    entropy_target =[]
    entropy_value = []
    ratios =[]

    for action, p, tp, r, b in zip(indices, probs, target_probs, reward, baseline):
        
        # ratio = torch.exp(p.log_prob(action) - tp.log_prob(action))
        # ratios.append(ratio)
        entropy_target.append(tp.entropy().detach().item())
        entropy_value.append(p.entropy().detach().item())
        
        advantage = r - b
        avg_advantage += advantage                                                                                        
        current_avg_advantage = advantage / len(indices)
        losses.append(-p.log_prob(action) * current_avg_advantage)

    abs_losses_rew = []
    abs_losses_pre = []

    for action, p, r, b in zip(act, act_probs, complex_rewards, act_baselines):
        advantage = r - b
        current_avg_advantage = advantage / len(act)      
        # Rewrite
        abs_losses_rew.append(-p.log_prob(action) * current_avg_advantage[0]) # divide by T*B
        # Preserve
        abs_losses_pre.append(-p.log_prob(action) * current_avg_advantage[1]) # divide by T*B

    critic_loss = F.mse_loss(baseline, reward)
    critic_loss_re = F.mse_loss(act_baselines, complex_rewards)

    # backprop and update
    autograd.backward(
      tensors=[critic_loss] + losses,
      grad_tensors=[torch.ones(1).to(critic_loss.device)]*(1+len(losses)),retain_graph=True)
    
    autograd.backward(
          tensors=[critic_loss_re] +abs_losses_rew + abs_losses_pre,
          grad_tensors=[torch.ones(1).to(critic_loss.device)]*(1+len(abs_losses_rew)+len(abs_losses_pre)))

    opt.step()
    target_agent.load_state_dict(agent.state_dict())
    ## clear cache
    torch.cuda.empty_cache()

    grad_log = grad_fn()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['entropy'] = sum(entropy_value)/len(entropy_value)
    log_dict['entropy_target'] = sum(entropy_target)/len(entropy_target)
    # log_dict['ratios'] = sum(ratios)/len(ratios)

    log_dict['reward'] = avg_reward / len(art_batch)
    log_dict['abs_reward_rew'] = avg_reward_abs_rew / len(art_batch)
    log_dict['abs_reward_pre'] = avg_reward_abs_pre / len(art_batch)

    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['act_mse'] = critic_loss_re.item()
    log_dict['recall'] = sum(recall)/len(recall)
    log_dict['precision'] = sum(precision)/len(precision)

    assert not math.isnan(log_dict['grad_norm'])

    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    
    params = [p for p in agent.parameters()]
    counts = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print('count:',counts)
    def f():
        grad_log = {}
        for n, m in agent.named_children():  
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            # if type(tot_grad) == float: break
            grad_log['grad_norm'+n] = tot_grad.item() 
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        # grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, target_net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._target_net = target_net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._target_net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing

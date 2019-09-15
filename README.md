
# Hybrid Net based Summarization with Cascading Agents
Here we supprt online demo and video demonstration. Also, in this webpage we describe some critical research results briefly. For more detail, please contact with me by github or go to our paper as following:
- [Link to googleDrive](https://drive.google.com/file/d/1mYhUS-zq_LUPwLV-pFa-zsBsFGgAj5v5/view?usp=sharing)

## Dependencies
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 1.1.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)

The code is tested on the *Linux* operating system.

## Evaluate the output summaries

To evaluate, you will need to download and setup the official ROUGE and METEOR
packages.

We use [`pyrouge`](https://github.com/bheinzerling/pyrouge)
(`pip install pyrouge` to install)
to make the ROUGE XML files required by the official perl script.
You will also need the official ROUGE package.
(However, it seems that the original ROUGE website is down.
An alternative can be found
*[here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)*.)
Please specify the path to your ROUGE package by setting the environment variable
`export ROUGE=[path/to/rouge/directory]`.

## Decode summaries from the pretrained model
Download the pretrained models *[here](https://drive.google.com/drive/folders/1TgS-Ug-BdtMwZh8up6uWDAxVlbwo908G?usp=sharing)*.
You will also need a preprocessed version of the CNN/DailyMail dataset.
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.
After that, specify the path of data files by setting the environment variable
`export DATA=[path/to/decompressed/data]`

To decode, run
```
python decode_full_model.py --path=[path/to/save/decoded/files] --model_dir=[path/to/pretrained] --beam=[beam_size] [--test/--val]
```
Options:
- beam_size: number of hypothesis for (diverse) beam search. (use beam_size > 1 to enable reranking)
  - beam_szie=1 to get greedy decoding results (rnn-ext + abs + RL)
  - beam_size=5 is used in the paper for the +rerank model (rnn-ext + abs + RL + rerank)
- test/val: decode on test/validation dataset

If you want to evaluate on the generated output files,
please follow the instructions in the above section to setup ROUGE/METEOR.

Next, make the reference files for evaluation:
```
python make_eval_references.py
```
and then run evaluation by:
```
python eval_full_model.py --[rouge/meteor] --decode_dir=[path/to/save/decoded/files]
```

### Results
You should get the following results

Test set

| Models             | ROUGEs (R-1, R-2, R-L) |
| ------------------ |:----------------------:|
| ext + abs          | (40.02, 17.53, 37.46)  |
| +absAgent 	     | (40.79, 18.53, 37.92)  |
| + rerank           | (41.39, 18.74, 38.51)  |

**NOTE**:
The original models in the paper are trained with pytorch 1.1.0 on python 3. 

## Train your own models
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.
After that, specify the path of data files by setting the environment variable
`export DATA=[path/to/decompressed/data]`

To re-train our best model:
1. pretrained a *word2vec* word embedding
```
python train_word2vec.py --path=[path/to/word2vec]
```
2. make the pseudo-labels
```
python make_extraction_labels.py
```
3. train *abstractor* and *extractor* using ML objectives
```
python train_abstractor.py --path=[path/to/abstractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
python train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
```
4. train the *full RL model*
```
python train_full_rl.py --path=[path/to/save/model] --abs_dir=[path/to/abstractor/model] --ext_dir=[path/to/extractor/model]
```
After the training finishes you will be able to run the decoding and evaluation following the instructions in the previous section.

The above will use the best hyper-parameters we used in the paper as default.
Please refer to the respective source code for options to set the hyper-parameters.

## Acknowledgement

Thanks to the open source provided from Yen-Chun Chen and Mohit Bansal that assists us to fininsh the whole research.
We appreciate and show the reference below: 
```
@inproceedings{chen2018fast,
  title={Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting},
  author={Yen-Chun Chen and Mohit Bansal},
  booktitle={Proceedings of ACL},
  year={2018}
}
```

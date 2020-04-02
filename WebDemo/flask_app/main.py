from flask import render_template, request
from flask_script import Manager, Server

from app import app
from model import Content, Summary, Article
import app.static.summ as summarizationModel
import os, json, logging

@app.route('/', endpoint='ACCESS')
@app.route('/index.html', endpoint='ACCESSFILE')
def index():
    try:
        all_pairs = Article.objects.all()
        return render_template('index.html', history=all_pairs)
    except Exception as e:
        logging.error(e)
        raise e

@app.route('/run_decode', methods=['POST'])
def run_decode():
    logging.debug('decode your input by our pretrained model')
    try:
        source = request.get_json()['source'] # GET request with String from frontend directly
        logging.debug('input: {}'.format(source)) # GET String-type context from the backend
        try:
            logging.debug('using the pretrained model.')
            sentNums, summary = summarizationModel.decode.run_(source)
        except Exception as e:
            logging.error(e)
        else:
            logging.debug('The number of sentences is {}'.format(sentNums))
            logging.debug('The abstract is that {}'.format(summary))
            results = {'sent_no': sentNums, 'final': summary}
            
        try:
            article = Content(text=source)
            abstract = Summary(text=summary)
            pair = Article(article=article.id, abstract=abstract.id)
            article.save()
            abstract.save()
            pair.save()
        except Exception as e:
            logging.error(e)

        return json.dumps(results)
    except:
        message = {'message' : 'Fail to catch the data from client.'}
        return json.dumps(message)

manager = Manager(app)
manager.add_command('runserver', Server(
    use_debugger = True,
    use_reloader = True,
    host = os.getenv('IP', '0.0.0.0'),
    port = int(os.getenv('PORT', 5001))
    ))
    
if __name__ == "__main__":
   manager.run()


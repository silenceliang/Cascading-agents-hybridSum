from flask import render_template, request
from flask_script import Manager, Server

from app import app
from model import Content, Summary, Article
import app.static.summ as summarizationModel
import os, json, logging

@app.route("/", endpoint='ACCESS')
@app.route("/index.html",endpoint='ACCESSFILE')
def index():
    all_pairs = Article.objects.all()
    return render_template('index.html', history=all_pairs)

@app.route("/run_decode", methods=['POST'])
def run_decode():
    logging.info('request is post to `run decode`')
    try:
        # GET request with type String FROM front-end directly
        source = request.get_json()['source']
        # GET RESULT FROM python script, return String type
        logging.debug('input: {}'.format(source))
        try:
            sentNums, summary = summarizationModel.decode.run_(source)
        except Exception as err:
            logging.error(err)
        logging.debug('output_sentNums: {}'.format(sentNums))
        logging.debug('output_summary: {}'.format(summary))

        results = {'sent_no': sentNums, 'final': summary}
        article = Content(text=source)
        abstract = Summary(text=summary)
        try:
            article.save()
            abstract.save()
        except:
            logging.error('Failed to save article & abstract')
        pair = Article(article=article.id, abstract=abstract.id)
        try:
            pair.save()
        except:
            logging.error('Failed to save article pair')           
        return json.dumps(results)
    except:
        message = {'message' : 'Fail to catch the data from client.'}
        return json.dumps(message)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

manager = Manager(app)
manager.add_command('runserver', Server(
    use_debugger = True,
    use_reloader = True,
    host = os.getenv('IP', '0.0.0.0'),
    port = int(os.getenv('PORT', 5001))
    ))
    
if __name__ == "__main__":
   manager.run()


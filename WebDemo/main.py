from flask import Flask, render_template, request
from flask_script import Manager, Server
from __init__ import app
from model import Content, Summary, Article
from static.summ import decode
import os, json, logging

manager = Manager(app)
manager.add_command('server', Server)

@app.route("/", endpoint='ACCESS')
@app.route("/index.html",endpoint='ACCESSFILE')
def index():
    all_pairs = Article.objects.all()
    print(all_pairs)
    return render_template('index.html', history=all_pairs)

@app.route("/run_decode", methods=['POST'])
def run_decode():
    try:
        # GET request with type String FROM front-end directly
        source = request.get_json()['source']
        # GET RESULT FROM python script, return String type
        sentNums, summary = decode.run_(source)
        results = {'sent_no': sentNums, 'final': summary}
        article = Content(text=source)
        abstract = Summary(text=summary)
        article.save()
        abstract.save()
        pair = Article(article=article.id, abstract=abstract.id)
        pair.save()
        return json.dumps(results)
    except:
        message = {'message' : 'Fail to catch the data from client.'}
        return json.dumps(message)


if __name__ == "__main__":
    manager.run()



from flask import Flask, render_template, request
from static.summ import decode
#from flask_pymongo import PyMongo
import os
import json

app = Flask(__name__,  static_url_path='', 
            static_folder='static',
            template_folder='templates')
#app.config['MONGO_URI'] = "mongodb://localhost:27017/myDatabase"
#mongo = PyMongo(app)

@app.route("/")
def index(): 
    return render_template('index.html')

@app.route("/index.html")
def index2():
    return render_template('index.html')

@app.route("/run_decode", methods=['POST'])
def getNotification():
    # GET VALUE FROM html script, Get String type
    source = request.get_json()['source']
    # GET RESULT FROM python script, return String type
    sentNums, summary = decode.run_(source)
 #   mongo.db.users.insert_one({"input":source, "output":summary})
    results = {"sent_no":sentNums,"final": summary}
    json_out = json.dumps(results)
    return json_out

# @app.route("/tables-basic.html")
# def tableBasic():
#     return render_template('tables-basic.html')


# @app.route("/ui-cards.html")
# def uiCards():
#     return render_template('ui-cards.html')

# @app.route("/font-themify.html")
# def fontThemify():
#     return render_template('font-themify.html')


# @app.route("/tables-data.html")
# def tablesData():
#     return render_template('tables-data.html')



# @app.route("/widgets.html")
# def widgets():
#     return render_template('widgets.html')


# @app.route("/ui-tabs.html")
# def uiTabs():
#     return render_template('ui-tabs.html')

# @app.route("/ui-modals.html")
# def uiModals():
#     return render_template('ui-modals.html')

# @app.route("/ui-progressbar.html")
# def uiProgressbar():
#     return render_template('ui-progressbar.html')

# @app.route("/ui-badges.html")
# def uiBadge():
#     return render_template('ui-badges.html')

# @app.route("/ui-alerts.html")
# def uiAlert():
#     return render_template('ui-alerts.html')


# @app.route("/ui-switches.html")
# def uiSwitches():
#     return render_template('ui-switches.html')

# @app.route("/ui-grids.html")
# def uiGrids():
#     return render_template('ui-grids.html')

# @app.route("/ui-typgraphy.html")
# def uiTypgraphy():
#     return render_template('ui-typgraphy.html')

# @app.route("/ui-buttons.html")
# def uiButton():
#     return render_template('ui-buttons.html')

# @app.route("/forms-basic.html")
# def formsBasic():
#     return render_template('forms-basic.html')


# @app.route("/charts-chartjs.html")
# def chartsChartjs():
#     return render_template('charts-chartjs.html')

# @app.route("/charts-flot.html")
# def chartsFlot():
#     return render_template('charts-flot.html')

# @app.route("/pages-forget.html")
# def pagesForget():
#     return render_template('pages-forget.html')

if __name__ == "__main__":
    app.run(host = "140.116.245.103", debug=True)


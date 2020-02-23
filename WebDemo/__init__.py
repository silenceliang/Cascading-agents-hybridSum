from flask import Flask
from flask_mongoengine import MongoEngine, MongoEngineSessionInterface

class Config(object):
    DEBUG = True
    TESTING = True

    MONGODB_SETTINGS = {
        'db': 'thesis',
        'host': '127.0.0.1',
        'port': 27017
    }
    DEBUG_TB_PANELS = (
        "flask_debugtoolbar.panels.versions.VersionDebugPanel",
        "flask_debugtoolbar.panels.timer.TimerDebugPanel",
        "flask_debugtoolbar.panels.headers.HeaderDebugPanel",
        "flask_debugtoolbar.panels.request_vars.RequestVarsDebugPanel",
        "flask_debugtoolbar.panels.template.TemplateDebugPanel",
        "flask_debugtoolbar.panels.logger.LoggingPanel",
        "flask_mongoengine.panels.MongoDebugPanel",
    )


app = Flask(__name__,  static_url_path='', 
            static_folder='static',
            template_folder='templates')    

app.config.from_object(Config)
db = MongoEngine(app)
app.session_interface = MongoEngineSessionInterface(db)

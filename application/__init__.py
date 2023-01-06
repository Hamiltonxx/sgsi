from flask import Flask 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from application.SGSI_pred import uploadTrain, evaluateSGSI, predictSGSI, downloadFile, progress
from application.hello import helloworld
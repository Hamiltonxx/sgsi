from flask import request 
from application import app 
import json

@app.route("/hello", methods=["GET","POST"])
def helloworld():
    if request.method == "POST":
        args = request.get_json(force=True)
        print(args)
        return json.dumps({"hello":"world"})
    else:
        return "Hello World"
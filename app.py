#web app
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("DBS_SGD.pkl") 

@app.route("/",methods=["GET","POST"])
def index():
  return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
  q=request.form.get("q")
  return(render_template("main.html"))

@app.route("/dbs",methods=["GET","POST"])
def dbs():
  return(render_template("dbs.html"))

@app.route("/dbsPred", methods=["GET","POST"])
def dbsPred():
  q = float(request.form.get("q"))
  X = pd.DataFrame([[q]], columns=["SGD"])  
  pred = model.predict(X)
  r = round(float(pred.ravel()[0]), 2)
  return(render_template("dbsPred.html", r=r))

if __name__ == "__main__" :
  app.run()

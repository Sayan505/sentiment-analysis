from engine import Engine

from flask import Flask, request, render_template




app = Flask(__name__)


# init prediction engine
model = Engine()


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")


@app.route("/api/predict_sentiment", methods=["POST"])
def predict_sentiment():
    req   = request.json

    query = str(req["query"])

    res   = model.predict(query)

    return res


@app.route("/api/status", methods=["GET"])
def get_status():
    return model.status

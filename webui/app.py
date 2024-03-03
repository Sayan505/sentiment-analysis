import engine

from flask import Flask, request, render_template, jsonify




app = Flask(__name__)


status = engine.load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")


@app.route("/api/predict_sentiment", methods=["POST"])
def predict_sentiment():
    req    = request.json

    query  = str(req["query"])

    
    result = str(engine.predict(query))

    return jsonify({ "prediction": result })


@app.route("/api/status", methods=["GET"])
def get_status():
    return status

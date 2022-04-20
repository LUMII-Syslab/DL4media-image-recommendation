import os, json
from flask import Flask, request, send_file

from code.code import get_images

app = Flask(__name__)
app.config["SECRET_KEY"] = "model!"


@app.route('/')
def get_index():
    return "Server for model is running"


# @app.route('/search-images/<text>')
@app.route('/search-images', methods=["POST"])
def extract_ticker():
    text = request.get_data().decode()
    
    print ("Searching ", text)

    idx = get_images(text)
    
    return send_file('images/' + str(idx[0]) + '.jpg')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', threaded=True)

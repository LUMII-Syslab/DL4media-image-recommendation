import os, json
from flask import Flask, request, send_file, render_template

from code.code import get_images

app = Flask(__name__)
app.config["SECRET_KEY"] = "model!"


@app.route('/')
def get_index():
	return render_template('search.html')


@app.route('/search-images', methods=["POST"])
def search_images():
	text = request.form.get("text")

	print ("Searching:", text)

	results = get_images(text)
	return render_template('result.html', results=results)


@app.route('/image/<filename>')
def get_image(filename):
	path = os.path.join("./images", filename)
	return send_file(path)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', threaded=True)

from flask import Flask, request, render_template

from utils import get_result

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/result", methods=["POST"])
def result():
    form_data = request.form
    sentence = form_data['sentence']
    output = get_result(sentence)
    return render_template('result.html', result=output)


if __name__ == '__main__':
    app.run(debug=True)

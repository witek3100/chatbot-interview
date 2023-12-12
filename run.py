from flask import Flask, render_template, request
from src.chatbot import generate_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('src/index.html')

@app.route('/ask', methods=['POST'])
def ask():
    pass

if __name__ == '__main__':
    app.run(debug=True)

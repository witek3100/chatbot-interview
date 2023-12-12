from flask import Flask, render_template, request, jsonify
import json
from src.indexing import create_pinecone_index
import openai
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


app = Flask(__name__)

with open('config.json') as config:
    config = json.load(config)

docsearch = create_pinecone_index()

def get_response(query):
    docs = docsearch.similarity_search(query)
    template = PromptTemplate.from_template(
        "You are given a texts and a query. Texts are transcriptions from video. You need to answer the query on the basis of paragraph. If paragraph doesn't contain relevant information your answer should be \"This interview doesn't contain this information \" \n\n Paragraph:\n{info} \n Query:\n {query}"
    )
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=config['open_ai_api_key'])
    chain = LLMChain(
        llm=llm,
        prompt=template,
        verbose=True
    )
    response = chain.predict(query=query, info='\n'.join([doc.page_content for doc in docs]))

    return response

@app.route('/', methods=['GET', 'POST'])
def run():
    if request.method == 'POST':
        query = request.form.get('query')
        result = get_response(query)
        response = {
            'result': result,
            'query': query
        }
        print(response)
        return jsonify(response)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

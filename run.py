from flask import Flask, render_template, request, jsonify
from src.indexing import create_pinecone_index
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os


app = Flask(__name__)
docsearch = create_pinecone_index()

def get_response(query):
    docs = docsearch.similarity_search(query)
    print(docs)
    template = PromptTemplate.from_template(
        "You are given a texts and a query. Texts are trasncriptions of youtube interview video."
        " You need to answer the query on the basis of texts. Try to answer only based on these texts, "
        "if it's impossible, answer something like that: \"This interview doesn't contain this information \" \n\n Paragraph:\n{info} \n Query:\n {query}"
    )
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1.5, openai_api_key=os.environ['OPENAI_API_KEY'])
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
        return jsonify(response)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)

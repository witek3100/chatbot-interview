import os
from flask import Flask, render_template, request, jsonify
from src.indexing import create_pinecone_index
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import json


app = Flask(__name__)
docsearch = create_pinecone_index()
with open('config.json') as config:
    config = json.load(config)

def get_response(query):
    docs = docsearch.similarity_search(query)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="human_input"
    )

    prompt = PromptTemplate(
        input_variables=[
            'query',
            'info',
            'chat_history',
            'human_input'
        ],
        template=(
            """
            "You are given a texts and a query. You need to answer the query on the basis of texts. Try to answer only based on these texts, "
            "if it's impossible, answer something like that: \"This interview doesn't contain this information \" "
            "Texts are transcripts form video interview video so you can thinks of terms video and interview interchangeably 
            \n Texts:\n{info}
            \n Query:\n{query}
            \n Previous conversation: {chat_history}"
            {human_input}
                """
        )
    )

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1.3, openai_api_key=os.environ['OPENAI_API_KEY'])
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    response = chain.predict(
        query=query,
        info='\n'.join([doc.page_content for doc in docs]),
        human_input=query
    )

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

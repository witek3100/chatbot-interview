import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


with open('../data/transcription.txt', 'r') as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = splitter.split_text(str(text))

with open('../config.json', 'r') as config:
    config = json.load(config)

embeddings = OpenAIEmbeddings(openai_api_key=config['open_ai_api_key'])
pinecone.init(
    api_key=config['pinecone_api_key'],
    environment=config['pinecone_enviroinment']
)
docsearch = Pinecone.from_texts([text for text in texts], embeddings, index_name='recru-task-chatbot')

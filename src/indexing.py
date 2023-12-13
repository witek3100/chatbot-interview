import os
import pinecone
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_pinecone_index():
    with open('data/transcription.txt', 'r') as file:
        text = file.read()

    with open('config.json') as config:
        config = json.load(config)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    texts = splitter.split_text(str(text))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_ENVIROINMENT']
    )
    docsearch = Pinecone.from_texts([text for text in texts], embeddings, index_name=os.environ['PINECONE_INDEX'])

    return docsearch

if __name__ == "__main__":
    create_pinecone_index()
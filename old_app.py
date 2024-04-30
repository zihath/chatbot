from PyPDF2 import PdfReader
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import spacy
from openai import OpenAI as openai_model

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

open_key = "sk-proj-Bb3NBdWGwg0c4CLtjf5RT3BlbkFJ7XMFZG39FPUCaHmkQiHE"

#Extract data from the PDF
with open("data.pickle", "rb") as f:
    text = pickle.load(f)


embeddings = OpenAIEmbeddings(openai_api_key = open_key)
docsearch = FAISS.from_texts(text, embeddings)
chain = load_qa_chain(OpenAI(), chain_type="stuff")

nlp = spacy.load("en_core_web_sm") 

def identify_question_type(query):
    doc = nlp(query)
    pdf_entities = set(ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE"))
    user_entities = set(ent.text for ent in doc.ents)

    # Handle empty user_entities set
    if not user_entities:
        return "GENERAL"  # Or a default value based on your logic

    overlap_ratio = len(pdf_entities.intersection(user_entities)) / len(user_entities)
    if overlap_ratio > 0.5:
        return "PDF_RELATED"
    return "GENERAL"



client = openai_model(
  api_key = open_key,
)


def get_response(query):
    question_type = identify_question_type(query)

    if question_type == "PDF_RELATED":
        docs = docsearch.similarity_search(query)
        response = chain.run(input_documents=docs, question= query)
    else:
        
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=50,
        messages=[
            {"role": "user", "content": query}
        ]
        )
        response = response.choices[0].message.content

    print(response)

    return response




# def main():
#     if 'responses' not in st.session_state:
#         st.session_state.responses = []

    
#     st.set_page_config("Chat with Anna ")
#     st.header("Anna university ChatBot")

#     user_question = st.text_input("Ask a Question about anna university")

#     if user_question:
#         response  = get_response(user_question)

#         st.session_state.responses.append("You: " + user_question)
#         st.session_state.responses.append("bot: " + response)

#         st.session_state.input = ""
#     for message in st.session_state.responses:
#         st.text_area("", message, height=20, disabled=True)


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return  get_response(input)


if __name__ == "__main__":
    app.run()
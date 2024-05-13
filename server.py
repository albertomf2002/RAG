import os
from flask import Flask, request
from flask import jsonify

from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False)

raw_prompt = PromptTemplate.from_template(""" 
    <s> [INST] 
        Eres un asistente enfocado al cuidado de personas mayores. 
        La respuesta debe ser siempre en español. 
        Nunca hagas introducción o te dirijas al usuario, responde directamente a la pregunta o frase que te formulen.
        Te he proporcionado un archivo denominado time.txt para que sepas la hora actual, es importante que única y exclusivamente tengas en cuenta la hora actual del último archivo time.txt que se te ha proporcionado, ningún otro.
        Si no puedes proporcionar una respuesta sacada de la información proporcionada, dilo.
    [/INST] </s>
                                          
    [INST]  {input} 
            Context: {context}
    [/INST]                                
""")

@app.route("/ask", methods=["POST"])
def askPost():
    print("Post /ask called")
    json_content = request.json
    query = json_content.get("query")

    vector_store = Chroma(persist_directory="db", embedding_function=embedding)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 20, "score_threshold": 0.1})
    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    response = chain.invoke({"input": query})
    response_answer = response["answer"]

    return jsonify(response_answer)

@app.route("/docs", methods=["POST"])
def docPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "docs/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = TextLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="db")

    response = {"status": "Succesfully Uploaded", "filename": file_name, "doc_len": len(docs), "chucks": len(chunks)}
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os

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
        Trata de hacer una respuesta sencilla y concisa con la información solicitada.
        En las preguntes que te haga, te paso la hora actual solo para que lo tengas en cuenta.
        Te he proporcionado un archivo denominado medicamentos.txt con el nombre de los medicamentos que me tengo que tomar y las horas de toma.
        Si no puedes proporcionar una respuesta sacada de la información proporcionada, dilo.
    [/INST] </s>
                                          
    [INST]  {input} 
            Context: {context}
    [/INST]                                
""")

# Historial de conversación
conversation_history = []

@app.route("/ask", methods=["POST"])
def askPost():
    print("Post /ask called")
    json_content = request.json
    query = json_content.get("query")

    try:
        # Actualizar historial de conversación
        conversation_history.append(f"Usuario: {query}")

        if "medicamento" in query.lower() or "medicina" in query.lower():
            vector_store = Chroma(persist_directory="db", embedding_function=embedding)
            retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 20, "score_threshold": 0.1})
            document_chain = create_stuff_documents_chain(llm, raw_prompt)
            chain = create_retrieval_chain(retriever, document_chain)
            response = chain.invoke({"input": query})
            response_answer = response["answer"]
        else:
            # Generar respuesta sin RAG
            open_prompt = f"""
            <s> [INST] 
                La respuesta debe ser siempre en español. 
                Nunca hagas introducción o te dirijas al usuario, responde directamente a la pregunta o frase que te formulen.
                Trata de hacer una respuesta corta, sencilla y concisa con la información solicitada.
                En las preguntes que te haga, te paso la hora actual solo para que lo tengas en cuenta.
                Aquí está el historial de la conversación hasta ahora:
                {' '.join(conversation_history)}
            [/INST] </s>
                                                  
            [INST]  {query}
            [/INST]
            """
            
            response_answer = llm.invoke(open_prompt)

        # Actualizar historial de conversación con la respuesta
        conversation_history.append(f"Asistente: {response_answer}")

        return jsonify({"answer": response_answer})

    except Exception as e:
        print(f"Error al generar respuesta: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/docs", methods=["POST"])
def docPost():
    print("Post /docs called")
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

@app.route("/informe", methods=["POST"])
def generate_adherence_report():
    log_tomas_path = 'docs/log_tomas.txt'
    log_estado_path = 'docs/log_estado.txt'
    report_tomas_path = 'docs/informe_tomas.txt'
    report_estado_path = 'docs/informe_estado.txt'

    if not os.path.exists(log_tomas_path):
        return jsonify({"status": "error", "message": "log_tomas.txt not found"}), 404

    if not os.path.exists(log_estado_path):
        return jsonify({"status": "error", "message": "log_estado.txt not found"}), 404

    adherence_data = {}
    total_medications = 0
    confirmed_takes = 0

    try:
        # Generar informe cuantitativo
        with open(log_tomas_path, 'r', encoding='utf-8') as log_file:
            lines = log_file.readlines()

        for line in lines:
            if "Es hora de tomar su medicamento:" in line:
                total_medications += 1
                med_name = line.split("Es hora de tomar su medicamento:")[1].split('(')[0].strip()
                if med_name not in adherence_data:
                    adherence_data[med_name] = {"total": 0, "confirmed": 0}
                adherence_data[med_name]["total"] += 1
            elif "El usuario ha confirmado la toma de" in line:
                confirmed_takes += 1
                med_name = line.split("El usuario ha confirmado la toma de")[1].strip().replace('.', '')
                if med_name in adherence_data:
                    adherence_data[med_name]["confirmed"] += 1

        adherence_percentage = (confirmed_takes / total_medications) * 100 if total_medications > 0 else 0

        report_lines = [
            "Informe de Adherencia del Paciente",
            "-----------------------------------",
            f"Total de medicamentos programados: {total_medications}",
            f"Total de medicamentos confirmados: {confirmed_takes}",
            f"Porcentaje de adherencia: {adherence_percentage:.2f}%",
            "\nAdherencia por medicamento:",
        ]

        for med_name, data in adherence_data.items():
            med_adherence = (data["confirmed"] / data["total"]) * 100 if data["total"] > 0 else 0
            report_lines.append(f"{med_name}: {med_adherence:.2f}% (Confirmados: {data['confirmed']}, Total: {data['total']})")

        with open(report_tomas_path, 'w', encoding='utf-8') as report_file:
            report_file.write("\n".join(report_lines))

        # Generar informe cualitativo
        with open(log_estado_path, 'r', encoding='utf-8') as log_file:
            estado_lines = log_file.readlines()

        estado_context = " ".join(estado_lines)
        qualitative_prompt = f"""
        <s> [INST] 
            Eres un asistente enfocado al cuidado de personas mayores. 
            La respuesta debe ser siempre en español. 
            Necesito que generes un informe cualitativo del estado de ánimo del paciente basado en los siguientes registros:
            {estado_context}
        [/INST] </s>
        """

        qualitative_response = llm.invoke(qualitative_prompt)
        qualitative_report = qualitative_response if isinstance(qualitative_response, str) else qualitative_response["text"]

        with open(report_estado_path, 'w', encoding='utf-8') as report_file:
            report_file.write(qualitative_report)

        return jsonify({"status": "success", "message": "Reports generated successfully", "report_tomas_file": "informe_tomas.txt", "report_estado_file": "informe_estado.txt"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()

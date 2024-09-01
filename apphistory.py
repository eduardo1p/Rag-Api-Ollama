from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

app = Flask(__name__)

historico_conversas = []

caminho_pasta = "db"

modelo_cache = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

divisor_texto = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

prompt_bruto = PromptTemplate.from_template(
    """ 
    <s>[INST] Você é um assistente técnico bom em pesquisar documentos. Se você não tiver uma resposta com base nas informações fornecidas, diga isso. [/INST] </s>
    [INST] {input}
           Contexto: {context}
           Resposta:
    [/INST]
"""
)


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai chamado")
    conteudo_json = request.json
    consulta = conteudo_json.get("query")

    print(f"Consulta: {consulta}")

    resposta = modelo_cache.invoke(consulta)

    print(resposta)

    resposta_final = {"resposta": resposta}
    return resposta_final


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf chamado")
    conteudo_json = request.json
    consulta = conteudo_json.get("query")

    print(f"Consulta: {consulta}")

    print("Carregando armazenamento vetorial")
    armazenamento_vetorial = Chroma(persist_directory=caminho_pasta, embedding_function=embedding)

    print("Criando cadeia")
    recuperador = armazenamento_vetorial.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    prompt_recuperador = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="historico_conversas"),
            ("human", "{input}"),
            (
                "human",
                "Dada a conversa acima, gere uma consulta de pesquisa para obter informações relevantes para a conversa",
            ),
        ]
    )

    recuperador_com_historico = create_history_aware_retriever(
        llm=modelo_cache, retriever=recuperador, prompt=prompt_recuperador
    )

    cadeia_documentos = create_stuff_documents_chain(modelo_cache, prompt_bruto)
    # cadeia = create_retrieval_chain(recuperador, cadeia_documentos)

    cadeia_recuperacao = create_retrieval_chain(
        # recuperador,
        recuperador_com_historico,
        cadeia_documentos,
    )

    # resultado = cadeia.invoke({"input": consulta})
    resultado = cadeia_recuperacao.invoke({"input": consulta})
    print(resultado["answer"])
    historico_conversas.append(HumanMessage(content=consulta))
    historico_conversas.append(AIMessage(content=resultado["answer"]))

    fontes = []
    for doc in resultado["context"]:
        fontes.append(
            {"fonte": doc.metadata["source"], "conteudo_pagina": doc.page_content}
        )

    resposta_final = {"resposta": resultado["answer"], "fontes": fontes}
    return resposta_final


@app.route("/pdf", methods=["POST"])
def pdfPost():
    arquivo = request.files["file"]
    nome_arquivo = arquivo.filename
    caminho_salvar = "pdf/" + nome_arquivo
    arquivo.save(caminho_salvar)
    print(f"Nome do arquivo: {nome_arquivo}")

    carregador = PDFPlumberLoader(caminho_salvar)
    documentos = carregador.load_and_split()
    print(f"Tamanho dos documentos={len(documentos)}")

    pedaços = divisor_texto.split_documents(documentos)
    print(f"Tamanho dos pedaços={len(pedaços)}")

    armazenamento_vetorial = Chroma.from_documents(
        documents=pedaços, embedding=embedding, persist_directory=caminho_pasta
    )

    armazenamento_vetorial.persist()

    resposta = {
        "status": "Enviado com sucesso",
        "nome_arquivo": nome_arquivo,
        "tamanho_documento": len(documentos),
        "pedaços": len(pedaços),
    }
    return resposta


def iniciar_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    iniciar_app()

# Api Rag

## Descrição

Este projeto utiliza Flask e LangChain para processar PDFs e realizar consultas inteligentes baseadas em contexto. Com ele, você pode enviar PDFs, extrair informações e fazer perguntas ao modelo de linguagem que levará em conta o conteúdo dos documentos e o histórico de conversas (na versão avançada).

## Instalação

Clone o repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

Instale as dependências:

pip install -r requirements.txt

Execute a aplicação:

Para a versão básica, execute:

python app.py

Para a versão com histórico de conversas, execute:

python apphistory.py

## Como Usar

Para enviar um PDF, use o comando:

curl -X POST -F 'file=@/caminho/para/seu/arquivo.pdf' http://0.0.0.0:8080/pdf

Para consultar PDFs, envie uma requisição POST com JSON:

curl -X POST -H "Content-Type: application/json" -d '{"query": "Sua pergunta"}' http://0.0.0.0:8080/ask_pdf

Para fazer uma pergunta ao modelo, use o comando:

curl -X POST -H "Content-Type: application/json" -d '{"query": "Sua pergunta"}' http://0.0.0.0:8080/ai

## Personalização

Você pode ajustar o modelo, o prompt e a forma como os PDFs são processados diretamente no código para atender às suas necessidades.

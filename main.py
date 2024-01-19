from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS
import os
import streamlit as st
import pickle
import time


from dotenv import load_dotenv
load_dotenv()

st.title("news")
st.sidebar.title("urls")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.2, max_tokens=200)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading..started..")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",".",","],chunk_size=1000)
    main_placeholder.text("text_splitter..started..")
    docs=text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("embedding vector started building...")
    with open(file_path,'wb') as f:
        pickle.dump(vectorstore_openai,f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vectorstore=pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"Question":query},return_only_outputs=True)
            # result has answer and sources
            st.header("Answer")
            st.subheader(result["answer"])
            #Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n") # split source by new line
                for source in sources_list:
                    st.wrte(source)





import streamlit as st
from math import ceil
import tempfile
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

st.title('🦜 VNCR-GPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key!',type="password")
uploaded_file_pdf = st.sidebar.file_uploader("Upload PDF Files",type=["pdf"],accept_multiple_files=True)
embeddings = OpenAIEmbeddings()

def generate_response(input_text):
  llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
  st.info(llm(str(input_text)))

def generate_response2(input_text):
  out=pdf_qa({"query": str(input_text)})
  res=out['result']
  ref=''.join(["\n "+ "Source: \n" + i.metadata['source'] +"\n Content"+ i.page_content for i in out['source_documents']])
  st.info(res+' \n \n '+"Reference: \n "+ref)
  

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Ask Me Anything')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if len(uploaded_file_pdf) != 0:
    docs=[]
    for uploadfile in uploaded_file_pdf:
      with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploadfile.getvalue())
        tmp_file_path = tmp_file.name
      pdf = PyPDFLoader(tmp_file_path)
      pages= pdf.load()
      if(len(pages)<=120):
        chunk=10000
      else:
        chunk=min(ceil(61800*(len(pages)/1000)),61800)
      text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk, chunk_overlap = 0)
      texts = text_splitter.split_documents(pages)
      for i in texts:
        i.metadata['source']=uploadfile.name
      docs.extend(texts)
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})
    pdf_qa= RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.2,model_name='gpt-3.5-turbo-16k'), chain_type="stuff", retriever=retriever, return_source_documents=True)
  else:
    try:
      vectordb.delete_collection()
    except:
      pass
  if len(uploaded_file_pdf) != 0 and submitted and openai_api_key.startswith('sk-'):
    generate_response2(text)
  if len(uploaded_file_pdf) == 0 and submitted and openai_api_key.startswith('sk-'):
    generate_response(text)

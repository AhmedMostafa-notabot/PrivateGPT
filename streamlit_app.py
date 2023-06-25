import streamlit as st
from math import ceil
import tempfile
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.title('🦜 VNCR-GPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key!',type="password")
uploaded_file_pdf = st.sidebar.file_uploader("Upload PDF Files",type=["pdf"])
def generate_response(input_text):
  llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
  st.info(llm(str(input_text)))

def generate_response2(input_text):
  embeddings = OpenAIEmbeddings()
  vectordb = FAISS.from_documents(texts, embedding=embeddings)
  topk=vectordb.similarity_search(str(input_text),k=1)
  pdf_qa = load_qa_chain(llm=OpenAI(temperature=0.2,model_name='gpt-3.5-turbo-16k'), chain_type="refine")
  out=pdf_qa.run(input_documents=topk, question=str(input_text))
  st.info(out)
  

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Ask Me Anything')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if uploaded_file_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_file.write(uploaded_file_pdf.getvalue())
      tmp_file_path = tmp_file.name
    pdf = PyPDFLoader(tmp_file_path)
    pages=pdf.load()
    if(len(pages)<=120):
      chunk=10000
    else:
      chunk=min(ceil(61800*(len(pages)/1000)),61800)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk, chunk_overlap = 0)
    texts = text_splitter.split_documents(pages)
  else:
    try:
      vectordb.delete_collection()
    except:
      pass
  if uploaded_file_pdf is not None and submitted and openai_api_key.startswith('sk-'):
    generate_response2(text)
  if uploaded_file_pdf is None and submitted and openai_api_key.startswith('sk-'):
    generate_response(text)

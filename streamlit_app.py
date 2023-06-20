import streamlit as st
# from langchain.llms import GPT4All
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI

st.title('🦜 VNCR-GPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key!',type="password")
uploaded_file_pdf = st.sidebar.file_uploader("Upload PDF Files",type=["pdf"], accept_multiple_files=True)
finished = st.sidebar.button('Remove PDF')
# col1, col2 = st.columns(2)
def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#   llm = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
  st.info(llm(str(input_text)))
def generate_response2(input_text):
  print(input_text)
#   llm = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
  st.info(pdf_qa({'question': str(input_text)})['answer'])

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Ask Me Anything')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if len(uploaded_file_pdf)!=0:
    text=[]
    pdf = PdfReader(uploaded_file_pdf[0])
    pages=pdf.pages
    for i in pages:
      text.append(i.extract_text())
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(text, embedding=embeddings, 
                                     persist_directory=".")
    vectordb.persist()
    memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True ,llm=OpenAI(temperature=0.7,model_name='gpt-3.5-turbo-16k'))
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.7,model_name='gpt-3.5-turbo-16k') , vectordb.as_retriever(),memory=memory)
  
  if len(uploaded_file_pdf)!=0 and submitted and openai_api_key.startswith('sk-'):
    generate_response2(text)
  if finished and submitted and openai_api_key.startswith('sk-'):
    uploaded_file_pdf=None
    generate_response(text)

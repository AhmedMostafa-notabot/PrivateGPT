import streamlit as st
import openai
# from langchain.llms import GPT4All
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI

st.title('🦜 VNCR-GPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key!',type="password")
uploaded_file_pdf = st.sidebar.file_uploader("Upload PDF Files",type=["pdf"])
# finished = st.sidebar.button('Remove PDF')
# col1, col2 = st.columns(2)
def generate_response(input_text):
  llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
#   llm = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
  st.info(llm(str(input_text)))
def generate_response2(input_text,history):
  # print(input_text)
#   llm = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
  out=pdf_qa({'question': str(input_text),'chat_history':history})['answer']
  st.info(out)
  history=[{str(input_text),out}]
  return history
  
def summarize_text(text):

  prompt =   [{"role": "user", "content": f"Summarize the following text in 5 sentences:\n{text}"}]
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=prompt,
      temperature=0.4, 
      max_tokens=150, # = 112 words
      top_p=0.9, 
      frequency_penalty=1,
      presence_penalty=0
  )

  return response["choices"][0]['message']['content']

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Ask Me Anything')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if uploaded_file_pdf is not None:
    # text=[]
    sumtext=[]
    pdf = PdfReader(uploaded_file_pdf)
    pages=pdf.pages
    for i in pages:
      # text.append(i.extract_text())
      sumtext.append(summarize_text(i.extract_text()))
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(sumtext, embedding=embeddings, 
                                     persist_directory=".")
    # vectordb.persist()
    chat_history=[]
    # memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True ,llm=OpenAI(temperature=0.4,model_name='gpt-3.5-turbo-16k'))
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.4,model_name='gpt-3.5-turbo-16k') , vectordb.as_retriever(search_type='similarity',search_kwargs={"k":5}))
  else:
    try:
      vectordb.delete_collection()
      chat_history=[]
    except:
      pass
  if uploaded_file_pdf is not None and submitted and openai_api_key.startswith('sk-'):
    chat_history=generate_response2(text,chat_history)
  if uploaded_file_pdf is None and submitted and openai_api_key.startswith('sk-'):
    generate_response(text)

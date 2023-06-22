import streamlit as st
import openai
import re
# from langchain.llms import GPT4All
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.memory import ConversationTokenBufferMemory
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

def generate_response2(input_text):
  # print(input_text)
#   llm = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
  # out=pdf_qa({'question': str(input_text)})['answer']
  out=pdf_qa.run(str(input_text))
  st.info(out)
  # history=[{str(input_text),out}]
  # return history
  
def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def summarize_text(text):

  prompt =   [{"role": "user", "content": f"Summarize this in 4 sentences:\n{text}"}]
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=prompt,
      temperature=0.2, 
      max_tokens=135,
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
    text=[]
    sumtext=[]
    pdf = PdfReader(uploaded_file_pdf)
    pages=pdf.pages
    minstep=min(len(pages),30)
    for i in pages:
      text.append(preprocess(i.extract_text()))
    for i in range(0,len(text),minstep):
      sumtext.append(summarize_text(''.join(text[i:i+minstep])))
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(sumtext, embedding=embeddings, 
                                     persist_directory=".")
    # vectordb.persist()
    # chat_history=[]
    # memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True ,llm=OpenAI(temperature=0.4,model_name='gpt-3.5-turbo-16k'))
    pdf_qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0,model_name='gpt-3.5-turbo-16k',max_tokens=135,frequency_penalty=1,presence_penalty=0),
                                                   retriever=vectordb.as_retriever(search_type='similarity',search_kwargs={"k":1}), chain_type="stuff")
    # ,memory=ConversationBufferWindowMemory(memory_key="chat_history",k=1,return_messages=True)
  else:
    try:
      vectordb.delete_collection()
      # chat_history=[]
    except:
      pass
  if uploaded_file_pdf is not None and submitted and openai_api_key.startswith('sk-'):
    generate_response2(text)
  if uploaded_file_pdf is None and submitted and openai_api_key.startswith('sk-'):
    generate_response(text)

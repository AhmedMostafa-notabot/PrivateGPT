import streamlit as st
# import openai
# import re
# from langchain.llms import GPT4All
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationTokenBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
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
  embeddings = OpenAIEmbeddings()
  vectordb = Chroma.from_texts(sumtext, embedding=embeddings, 
                                     persist_directory=".").as_retriever()
  topk=vectordb.get_relevant_documents(str(input_text))
  # sumvectordb=Chroma.from_documents(topk,embedding=embeddings,persist_directory=".")
  # sumvectordb.persist()
    # chat_history=[]
    # memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True ,llm=OpenAI(temperature=0.4,model_name='gpt-3.5-turbo-16k'))
  # memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True ,llm=OpenAI(temperature=0.2,model_name='gpt-3.5-turbo-16k'))
  pdf_qa = load_qa_chain(llm=OpenAI(temperature=0.2,model_name='gpt-3.5-turbo-16k'), chain_type="stuff")
  out=pdf_qa.run(input_documents=topk, question=str(input_text))
  # print(input_text)
#   llm = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
  # out=pdf_qa({'question': str(input_text)})['answer']
  # out=pdf_qa.run(str(input_text))
  st.info(out)
  # history=[{str(input_text),out}]
  # return history
  
# def preprocess(text):
#     text = text.replace('\n', ' ')
#     text = re.sub('\s+', ' ', text)
#     return text

# def summarize_text(text):

#   prompt =   [{"role": "user", "content": f"Summarize this in 5 sentences:\n{text}"}]
#   response = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo-16k", 
#       messages=prompt,
#       temperature=0.2, 
#       max_tokens=100,
#       top_p=0.9
#   )

#   return response["choices"][0]['message']['content']

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Ask Me Anything')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if uploaded_file_pdf is not None:
    stext=[]
    sumtext=[]
    pdf = PdfReader(uploaded_file_pdf)
    pages=pdf.pages
    # minstep=min(len(pages),20)
    for i in pages:
      stext.append(i.extract_text())
    finaltext=''.join(stext)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = min(ceil(61800*(len(pages)/1000)),61800), chunk_overlap = 0)
    texts = text_splitter.create_documents([finaltext])
    chain = load_summarize_chain(llm=OpenAI(temperature=0,model_name='gpt-3.5-turbo-16k',frequency_penalty=1,presence_penalty=0), chain_type="refine", return_intermediate_steps=True)
    sumtext=chain(texts,return_only_outputs=True)['intermediate_steps']
    # print(sumtext)
    # for i in range(0,len(text),minstep):
    #   try:
    #     sumtext.append(summarize_text(''.join(text[i:i+minstep])))
    #   except:
    #     pass
    # ,memory=ConversationBufferWindowMemory(memory_key="chat_history",k=1,return_messages=True)
  else:
    try:
      vectordb.delete_collection()
      # sumvectordb.delete_collection()
      stext=None
      sumtext=None
    except:
      pass
  if uploaded_file_pdf is not None and submitted and openai_api_key.startswith('sk-'):
    generate_response2(text)
  if uploaded_file_pdf is None and submitted and openai_api_key.startswith('sk-'):
    generate_response(text)

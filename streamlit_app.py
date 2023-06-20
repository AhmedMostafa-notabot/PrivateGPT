import streamlit as st
# from langchain.llms import GPT4All
from langchain.llms import OpenAI

st.title('🦜 VNCR-GPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key!')
uploaded_file_pdf = st.sidebar.file_uploader("Upload PDF Files",type=["pdf"], accept_multiple_files=True)
def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#   llm = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
  st.info(llm(input_text))

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if uploaded_file_pdf:
    path=uploaded_file_pdf.read()
    print(path)
  if submitted and openai_api_key.startswith('sk-'):
    print(text)
    generate_response(text)

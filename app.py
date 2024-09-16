import streamlit as st
import os
from io import BytesIO
from typing import List
from multiprocessing import Pool
import tempfile
from tqdm import tqdm
from PIL import Image
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

######################### HTML CSS ############################
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1" style="max-height: 80px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
###################################################

chunk_size = 500
chunk_overlap = 50
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 5))

from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_content: BytesIO, file_type: str) -> List[Document]:
    ext = "." + file_type.rsplit("/", 1)[1]

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        temp_file.write(file_content.getvalue())
        temp_file_path = temp_file.name

    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(temp_file_path, **loader_args)
        results = loader.load()
        os.remove(temp_file_path)
        return results

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_uploaded_documents(uploaded_files, uploaded_files_type, ignored_files: List[str] = []) -> List[Document]:
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(uploaded_files), desc='Loading new documents', ncols=80) as progress_bar:
            for i, uploaded_file in enumerate(uploaded_files):
                file_type = uploaded_files_type[i]
                file_content = BytesIO(uploaded_file.read())
                docs = load_single_document(file_content, file_type)
                results.extend(docs)
                progress_bar.update()
    return results


def get_pdf_text(uploaded_files):
    ignored_files = []  # Add files to ignore if needed

    uploaded_files_list = [file for file in uploaded_files]
    uploaded_files_type = [file.type for file in uploaded_files]
    results = load_uploaded_documents(uploaded_files_list, uploaded_files_type, ignored_files)
    return results


def get_text_chunks(results, chunk_size, chunk_overlap):
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(results)
    return texts


def get_vectorstore(text_chunks, embeddings):
    # Use FAISS vectorstore
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, target_source_chunks):
    retriever = vectorstore.as_retriever(search_kwargs={"k": target_source_chunks})

    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token='Your API Key',
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain


st.set_page_config(page_title="Generate Insights", page_icon=":bar_chart:")


def handle_userinput(user_question):
    # Ensure conversation exists before using it
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("No conversation initialized. Please upload documents and process them first.")


def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo


st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col5:
    my_logo = add_logo(logo_path="attachment_73051350.jpeg", width=100, height=20)
    st.image(my_logo)
with col6:
    pg_logo = add_logo(logo_path="Q&A logo.jfif", width=60, height=40)
    st.image(pg_logo)


def main():
    css2 = '''
    <style>
        [data-testid="stSidebar"]{
            min-width: 300px;
            max-width: 300px;
        }
    </style>
    '''
    st.markdown(css2, unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(":blue Generate Insights :bar_chart:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload documents", type=["pdf", "xlsx", 'csv'], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):

                # get pdf text
                if uploaded_files is not None:
                    raw_text = get_pdf_text(uploaded_files=uploaded_files)

                    # get the text chunks
                    text_chunks = get_text_chunks(results=raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                    # create embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2"
                    )

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks=text_chunks, embeddings=embeddings)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore=vectorstore,
                                                                           target_source_chunks=target_source_chunks)


if __name__ == '__main__':
    main()

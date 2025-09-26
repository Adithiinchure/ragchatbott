import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="PDF Q&A with Groq", layout="wide")
st.title("📄 Get Instant Answers from Your PDF")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Step 1: Read PDF
    reader = PdfReader(uploaded_file)
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Step 2: Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    # Step 3: Embed and store in Chroma (force CPU to avoid meta tensor error)
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectordb = Chroma.from_texts(chunks, embedding=embedder, persist_directory="chroma_db")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Step 4: Setup LLM (Groq DeepSeek)
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b")

    # Step 5: Define custom prompt
    prompt_text = """You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Step 6: User input
    user_question = st.text_input("Ask a question about the PDF:")

    if st.button("Get Answer"):
        if user_question.strip():
            # Retrieve relevant chunks (use invoke instead of deprecated method)
            docs = retriever.invoke(user_question)
            context = "\n".join([doc.page_content for doc in docs])

            # Generate response
            response = chain.invoke({"context": context, "question": user_question})

            # Display answer
            st.subheader("Answer")
            st.write(response["text"])
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load .env keys
load_dotenv()

# Step 1: Read PDF
reader = PdfReader("Incorrect_facts.pdf.pdf")
full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Step 2: Chunk the text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)

# Step 3: Embed and create Chroma DB
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_texts(chunks, embedding=embedder, persist_directory="chroma_db")
vectordb.persist()

# Step 4: Setup LLM
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b")

# Step 5: Setup retriever with k=3
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Step 6: Define custom prompt
prompt_text = """You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_text)

# Step 7: Chain prompt with LLM
chain = LLMChain(llm=llm, prompt=prompt)

# Step 8: Ask multiple questions and combine results
while True:
    user_input = input("Ask your question(s) (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    # Handle multiple questions at once
    questions = [q.strip() + "?" for q in user_input.split("?") if q.strip()]

    for question in questions:
        # Retrieve top documents
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])

        # Generate response from LLM
        response = chain.invoke({"context": context, "question": question})
        print(f"\nQ: {question}\nA: {response['text']}")


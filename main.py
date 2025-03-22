from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import textwrap
import pytesseract
from pdf2image import convert_from_path
import traceback
import os

# Set the Tesseract executable path explicitly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

file_path = "./example_data/cia-astral-projection.pdf"

def extract_text_from_image_pdf(file_path):
    """
    Extracts text from an image-based PDF using OCR.
    Returns a list of Document objects.
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        # Verify Tesseract is accessible
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            raise FileNotFoundError(f"Tesseract executable not found at: {pytesseract.pytesseract.tesseract_cmd}")

        # Convert PDF to images (requires poppler installed)
        print("Converting PDF to images...")
        images = convert_from_path(file_path, dpi=200)  # Adjust DPI for quality
        
        ocr_docs = []
        for i, image in enumerate(images):
            print(f"Extracting text from page {i + 1}...")
            text = pytesseract.image_to_string(image, lang='eng')  # Specify language
            if text.strip():
                ocr_docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page_label": str(i + 1)}
                ))
            else:
                print(f"Warning: No text extracted from page {i + 1}")
        
        if not ocr_docs:
            print("No text extracted from the PDF.")
            return [Document(page_content="No text extracted from PDF", metadata={"source": file_path})]
        
        print(f"Successfully extracted text from {len(ocr_docs)} pages.")
        return ocr_docs
    
    except Exception as e:
        print(f"Error during text extraction: {str(e)}")
        traceback.print_exc()  # Print full stack trace for debugging
        return [Document(page_content=f"Error during extraction: {str(e)}", metadata={"source": file_path})]

# Extract text from the image-based PDF
docs = extract_text_from_image_pdf(file_path)

# Initialize language model
llm = ChatOpenAI(model="o3-mini-2025-01-31")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Set up retriever
retriever = vectorstore.as_retriever()

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don’t know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def format_rag_results(results):
    """
    Formats the RAG results with clear, readable output.
    """
    print("=" * 50)
    print("🔍 RAG QUERY RESULTS 🔍")
    print("=" * 50)
    
    print(f"\n📌 Question: {results['input']}")
    print("\n📝 Answer:")
    print(textwrap.fill(results['answer'], width=80))
    
    print("\n📚 References:")
    for i, doc in enumerate(results['context'], 1):
        print(f"\nReference {i}:")
        print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"  Page: {doc.metadata.get('page_label', 'N/A')}")
        print("  Excerpt:")
        print(textwrap.fill(doc.page_content[:300] + "...", width=70, initial_indent="    ", subsequent_indent="    "))
    
    print("\n" + "=" * 50)
    print("RAW DATA RESPONSE")
    print("=" * 50)

# Invoke the chain with your query
results = rag_chain.invoke({"input": "What is this document about? What practical advice could you give to achieve astral projection?"})
format_rag_results(results)

print(results)
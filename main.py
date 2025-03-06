from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import textwrap

load_dotenv()

file_path = "./example_data/Meditations_Marcus_Aurelius.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

llm = ChatOpenAI(model="gpt-4o")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
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

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def format_rag_results(results):
    """
    Formats the RAG results with clear, readable output.
    
    Args:
        results (dict): The results from the RAG chain
    """
    print("=" * 50)
    print("🔍 RAG QUERY RESULTS 🔍")
    print("=" * 50)
    
    # Print Question
    print(f"\n📌 Question: {results['input']}")
    
    # Print Answer
    print("\n📝 Answer:")
    print(textwrap.fill(results['answer'], width=80))
    
    # Print References
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

results = rag_chain.invoke({"input": "Dame un mantra para aceptar la parte tediosa y dificil de hacer algo grande, esos sacrificios que uno tiene que hacer... Ostinato Rigore"})
format_rag_results(results)

print(results)
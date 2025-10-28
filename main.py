import os
import glob
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

def setup_knowledge_base():
    """Load all documents and create the vector database"""
    
    # List to hold all documents
    all_documents = []
    
    # Load PDF files
    pdf_files = glob.glob("knowledge_base/*.pdf")
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"  - Loaded {len(documents)} pages")
        except Exception as e:
            print(f"  - Error loading {pdf_file}: {e}")
    
    # Load text files
    txt_files = glob.glob("knowledge_base/*.txt")
    for txt_file in txt_files:
        print(f"Loading text file: {txt_file}")
        try:
            loader = TextLoader(txt_file)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"  - Loaded {len(documents)} sections")
        except Exception as e:
            print(f"  - Error loading {txt_file}: {e}")
    
    if not all_documents:
        print("No documents found in knowledge_base folder!")
        print("Please add some PDF or text files to the knowledge_base folder.")
        return None
    
    # Split documents into chunks
    print(f"\nLoaded {len(all_documents)} total documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Create vector database
    print("Creating vector database...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Knowledge base setup complete!")
    return vector_db

def initialize_chatbot():
    """Initialize the chatbot with the knowledge base"""
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Please set your GOOGLE_API_KEY in the .env file")
        print("Get a free API key from: https://aistudio.google.com/")
        return None, None
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if we have an existing vector database
    if os.path.exists("./chroma_db"):
        print("Loading existing knowledge base...")
        try:
            vector_db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            # Test if it works
            _ = vector_db.similarity_search("test", k=1)
            print("Successfully loaded existing knowledge base!")
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print("Creating new knowledge base...")
            vector_db = setup_knowledge_base()
    else:
        print("Creating new knowledge base...")
        vector_db = setup_knowledge_base()
    
    return llm, vector_db

def ask_question(question, llm, vector_db):
    """Ask a question and get an answer"""
    if not vector_db:
        return "Knowledge base not available. Please check the setup."
    
    # Retrieve relevant documents
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    
    if not relevant_docs:
        return "I couldn't find any relevant information about that in my knowledge base. Please try asking differently or contact support."
    
    # Combine the context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create the prompt
    prompt = f"""You are a helpful customer support assistant. Answer the question based ONLY on the following context from our product documentation.

If the answer cannot be found in the context, say "I don't have enough information to answer that question. Please contact our support team for more help."

Context:
{context}

Question: {question}

Answer:"""
    
    # Get the response
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error getting response: {e}"

# Main execution
if __name__ == "__main__":
    print("🤖 Initializing Business Chatbot...")
    print("=" * 50)
    
    llm, vector_db = initialize_chatbot()
    
    if llm and vector_db:
        print("\n✅ Chatbot is ready! Ask me anything about your products.")
        print("💡 Type 'quit', 'exit', or 'bye' to end the conversation.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! 👋")
                    break
                if not question:
                    continue
                
                print("Bot: Thinking...")
                answer = ask_question(question, llm, vector_db)
                print(f"Bot: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    else:
        print("❌ Failed to initialize chatbot. Please check the errors above.")
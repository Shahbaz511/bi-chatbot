import os
import glob
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_knowledge_base():
    """Load all text files from knowledge_base folder"""
    knowledge_content = ""
    
    # Load text files
    txt_files = glob.glob("knowledge_base/*.txt")
    for txt_file in txt_files:
        print(f"Loading: {txt_file}")
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                knowledge_content += f"\n\n--- Content from {os.path.basename(txt_file)} ---\n"
                knowledge_content += file.read()
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    
    if not knowledge_content:
        print("No knowledge base files found! Please add .txt files to knowledge_base folder.")
        return None
    
    return knowledge_content

def initialize_chatbot():
    """Initialize the chatbot"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY in .env file")
        print("Get free API key from: https://aistudio.google.com/")
        return None, None
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Load knowledge base
    print("Loading knowledge base...")
    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return None, None
    
    # Initialize the model with correct model name
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    return model, knowledge_base

def ask_question(question, model, knowledge_base):
    """Ask a question using the knowledge base"""
    
    # Create the prompt with context
    prompt = f"""You are a helpful customer support assistant for Business Institute Bahrain, a Xerox solutions provider. Use the following business information to answer questions.

BUSINESS KNOWLEDGE:
{knowledge_base}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the business knowledge above
- Be helpful and friendly
- If the answer isn't in the knowledge, say: "I don't have that specific information. Please contact our support team at info@bi-bh.com or visit our website https://www.bi-bh.com/"
- Keep answers clear and concise
- Always represent Business Institute Bahrain professionally

ANSWER:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🤖 Business Institute Bahrain - Xerox Solutions Chatbot")
    print("=" * 55)
    
    model, knowledge_base = initialize_chatbot()
    
    if model and knowledge_base:
        print("\n✅ Chatbot ready! Ask me about our Xerox products and services.")
        print("💡 Type 'quit' to exit\n")
        
        while True:
            try:
                question = input("You: ").strip()
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! 👋")
                    break
                if not question:
                    continue
                
                print("Bot: Thinking...")
                answer = ask_question(question, model, knowledge_base)
                print(f"Bot: {answer}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    else:
        print("❌ Failed to initialize. Check errors above.")

if __name__ == "__main__":
    main()
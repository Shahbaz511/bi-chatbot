import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def test_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ No API key found in .env file")
        return
    
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-5:]}")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print("✅ API configuration successful")
        
        # List available models
        print("\n📋 Checking available models...")
        models = genai.list_models()
        
        print("✅ Available models:")
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"   - {model.name}")
        
        # Test a specific model
        print("\n🧪 Testing gemini-pro model...")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say 'Hello' in one word")
        print(f"✅ Model test successful: {response.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_key()
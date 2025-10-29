import gradio as gr
import os
import google.generativeai as genai

# Set your API key
api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY as a secret in Hugging Face Spaces")

genai.configure(api_key=api_key)

# Your business knowledge
business_knowledge = """
Business International Group

About Us:
Business International Group is a leading provider of Xerox document management solutions and services in Bahrain. We specialize in delivering cutting-edge document technology solutions to businesses across various sectors.

Our Services:
- Xerox Document Management Systems
- Printers and Multifunction Devices
- Digital Printing Solutions
- Document Workflow Automation
- Managed Print Services
- Technical Support and Maintenance

XEROX PRODUCTS:
- Xerox WorkCentre Multifunction Printers
- Xerox VersaLink Business Series
- Xerox AltaLink Multifunction Printers
- Xerox Phaser Printers
- Xerox ColorQube Inkjet Printers

Contact Information:
Website: https://www.bi-bh.com/
Email: info@bi-bh.com
Location: Bahrain

If you need specific pricing or technical details, please contact us directly at info@bi-bh.com
"""

def get_response(message, history):
    """Function to handle chat responses"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""You are a helpful customer support assistant for Business Institute Bahrain (BIBH). 
Use this business information to answer questions:

{business_knowledge}

Current conversation history: {history}
User question: {message}

Instructions:
- Answer based ONLY on the business information above
- Be helpful and professional
- If you don't have the specific information, suggest contacting info@bi-bh.com
- Keep answers clear and concise
- Be friendly and welcoming

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "I apologize, but I'm having trouble responding right now. Please contact us directly at info@bi-bh.com"

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=get_response,
    title="🤖 BIBH Xerox Solutions Assistant",
    description="Welcome to Business International Group! Ask me about our Xerox products, services, or support.",
    examples=[
        "What Xerox printers do you offer?",
        "Do you provide maintenance services?",
        "How can I contact your support team?",
        "What areas in Bahrain do you serve?"
    ],
    theme="soft"
)

if __name__ == "__main__":

    demo.launch(share=True)

import google.generativeai as genai
import os
from dotenv import load_dotenv  # Import this

# Load environment variables from .env file
load_dotenv() 

# Now this will actually get the key
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

    print("Available Embedding Models:")
    try:
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
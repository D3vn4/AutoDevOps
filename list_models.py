import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# List available models
models = genai.list_models()
print("Available models:")
for m in models:
    # Print model name and its capabilities
    print(m.name, "-", getattr(m, "capabilities", "N/A"))

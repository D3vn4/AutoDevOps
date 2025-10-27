import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
else:
    print("API Key loaded successfully.")

# Configure Gemini model
genai.configure(api_key=google_api_key)
# Use the model name confirmed from your list
model = genai.GenerativeModel("gemini-flash-latest")
print("Gemini Model configured.")

# Helper function to ask Gemini
def ask_gemini(prompt):
    """
    Sends a prompt to the Gemini model and returns the text response.

    Parameters:
        prompt (str): The text prompt to send to the model.

    Returns:
        str: The model's text response, or an error message if extraction fails.
    """
    print("Calling Gemini API...")
    try:
        response = model.generate_content(prompt)
        print("Gemini API call successful.")

        # Extract text from known response structures
        if hasattr(response, 'text'):
            print(" Extracted text using response.text")
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates and \
             hasattr(response.candidates[0], 'content') and \
             hasattr(response.candidates[0].content, 'parts') and \
             response.candidates[0].content.parts:
            print(" Extracted text using response.candidates structure")
            # Ensure the part actually contains text before accessing .text
            if hasattr(response.candidates[0].content.parts[0], 'text'):
                return response.candidates[0].content.parts[0].text.strip()
            else:
                 print(" Part found, but no text attribute inside the part.")
                 print(" Part details:", response.candidates[0].content.parts[0])
                 return "Error: Part structure incorrect."
        else:
            print("Could not find text in response using known methods.")
            print("Response details:", response)
            return "Error: Could not extract text from response structure."

    except Exception as e:
        print(f"Error during Gemini API call or processing: {e}")
        # print("Full Response on error:", response) # Uncomment for debugging if needed
        return f"Error: An exception occurred - {e}"

# --- NEW: Function to read code from a file ---
def read_code_from_file(file_path):
    """Reads the content of a specified file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # Added encoding
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # --- Specify the file you want to review ---
    file_to_review = "sample.py" # Use the folder name and the filename
    print(f"Reading code from: {file_to_review}")
    code_content = read_code_from_file(file_to_review)

    if code_content: # Only proceed if the file was read successfully
        # --- Define the prompt using the file content ---
        code_review_prompt = (
            "You are an expert Python code reviewer acting as a Senior Software Engineer.\n"
            "Your goal is to analyze Python code snippets for bugs, style issues (PEP 8),\n"
            "missing comments, and suggest improvements with corrected code examples.\n"
            "Provide clear, actionable feedback in markdown format.\n\n"
            "Focus on:\n"
            "- Potential runtime errors (like division by zero).\n"
            "- Adherence to PEP 8 style guidelines.\n"
            "- Clarity and necessity of comments.\n"
            "- Opportunities for improving efficiency or readability.\n"
            "- Security vulnerabilities (if any).\n\n"
            f"Please review the following Python code from the file '{file_to_review}':\n\n" # Added filename context
            "CODE:\n"
            "```python\n"
            f"{code_content}\n" # <<< USE THE FILE CONTENT HERE
            "```\n\n"
            "Output your review as a concise markdown report, including identified issues\n"
            "and a block showing the corrected/improved code if necessary."
        )

        # Get the code review
        print("🚀 Reviewing Code...")
        review = ask_gemini(code_review_prompt)

        print("\nCode Review Result:\n")
        print("-------------------------------")
        if review:
            print(review)
        else:
            print("The review content is empty or None.")
    else:
        print("Script finished because the file could not be read.")
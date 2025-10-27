import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
# No need for langchain or google libraries directly if using litellm string

# Load API key from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
else:
    # Set the environment variable specifically for litellm/crewai if needed, though load_dotenv usually suffices
    os.environ['GOOGLE_API_KEY'] = google_api_key
    print("✅ API Key loaded successfully.")

# --- LLM CONFIG IS NOW DONE INSIDE THE AGENT via string ---
print("✅ LLM will be configured within the Agent using litellm string.")

# --- Define a Custom Tool for Reading Files ---
class FileReaderTool(BaseTool):
    name: str = "File Reader Tool"
    description: str = "Reads the content of a specified file path."

    def _run(self, file_path: str) -> str:
        """Reads the content of the file."""
        try:
            safe_base_path = os.path.abspath(".")
            target_path = os.path.abspath(file_path)
            if not target_path.startswith(safe_base_path):
                return f"Error: Access denied. Cannot read files outside the project directory."

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            return f"Error: File not found at '{file_path}'"
        except Exception as e:
            return f"Error reading file '{file_path}': {e}"

# Instantiate the tool
file_reader_tool = FileReaderTool()
print("✅ File Reader Tool created.")

# --- Define the Code Reviewer Agent ---
code_reviewer = Agent(
    role='Expert Python Code Reviewer',
    goal='Analyze Python code read from a file for bugs, style issues (PEP 8), missing comments, and suggest improvements with corrected code examples.',
    backstory=(
        "You are a meticulous Senior Software Engineer known for your deep understanding of Python best practices, "
        "code efficiency, and security vulnerabilities. You use available tools to read files and provide clear, actionable feedback."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[file_reader_tool],
    # --- Pass the litellm-compatible string ---
    llm="gemini/gemini-flash-latest"
    # CrewAI/LiteLLM will use the GOOGLE_API_KEY environment variable
)
print("✅ Code Reviewer Agent defined.")

# --- Define the Review Task ---
file_to_review = "sample.py" # Ensure this path is correct

review_task = Task(
    description=(
        f"Use the 'File Reader Tool' to read the content of the file located at '{file_to_review}'. "
        f"Then, meticulously review the Python code obtained from the file. Identify any bugs, "
        f"check for PEP 8 style violations, suggest missing comments, analyze potential performance issues (like blocking I/O), "
        f"and provide an improved version of the code if necessary."
    ),
    expected_output=(
        "A concise markdown report identifying issues found (bugs, style, comments, performance) "
        "based *only* on the content read from the file. Include a block showing the corrected/improved code "
        "if applicable. If the file cannot be read, report the error provided by the tool."
    ),
    agent=code_reviewer,
)
print(f"✅ Review Task defined for file: {file_to_review}")

# --- Create and Run the Crew ---
code_review_crew = Crew(
    agents=[code_reviewer],
    tasks=[review_task],
    process=Process.sequential,
    verbose=True
)
print("✅ Crew created.")

print("\n🚀 Starting Code Review Crew...")
result = code_review_crew.kickoff()

print("\n\n✅ Crew Execution Finished!")
print("Code Review Result:")
print("-------------------------------")
print(result)
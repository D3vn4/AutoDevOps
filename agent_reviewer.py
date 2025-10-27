"""
AutoDevOps AI Agent Crew
------------------------
This script launches a multi-agent crew to automate the Pull Request (PR)
code review process.

The crew consists of four agents:
1.  **Code Reviewer**: Reads .py files from a GitHub PR and provides a code review.
2.  **Test Generator**: Writes pytest unit tests based on the reviewer's corrected code.
3.  **Test Executor**: Runs the generated tests in a secure environment.
4.  **Report Agent**: Posts the final review and test summary to the GitHub PR.

This script requires:
- A .env file with GOOGLE_API_KEY and GITHUB_PAT.
- Installed libraries: crewai, crewai-tools, python-dotenv, PyGithub, pytest, litellm.
"""

# --- 1. Imports ---

# Standard Library
import os
import sys
import subprocess
import tempfile
import logging

# Third-Party Libraries
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from github import Github

# --- 2. Configuration and Setup ---

# Configure logging for clear, professional-looking output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Update this URL to point to the Pull Request you want to review
PR_TO_REVIEW = "https://github.com/D3vn4/AutoDevOps/pull/1"
LLM_MODEL = "gemini/gemini-flash-latest"

def setup_environment():
    """
    Loads API keys from the .env file, validates them,
    and sets the GOOGLE_API_KEY environment variable for CrewAI/LiteLLM.
    Returns the GitHub PAT for use in the GitHub tool.
    """
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    github_pat = os.getenv("GITHUB_PAT")

    if not google_api_key:
        logging.error("GOOGLE_API_KEY not found in .env file.")
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    if not github_pat:
        logging.error("GITHUB_PAT not found in .env file.")
        raise ValueError("GITHUB_PAT not found in .env file")

    # Set the Google API key in the environment for LiteLLM to auto-detect
    os.environ['GOOGLE_API_KEY'] = google_api_key
    logging.info("API Keys loaded successfully.")
    return github_pat

# --- 3. Custom Agent Tools ---

class GitHubPRTool(BaseTool):
    """
    A custom tool for reading all .py files from a GitHub Pull Request.
    This tool authenticates using a GitHub Personal Access Token (PAT).
    """
    name: str = "GitHub PR File Reader"
    description: str = "Reads all .py files from a given GitHub Pull Request URL."
    github_client: Github = None

    def __init__(self, github_pat: str):
        """
        Initializes the tool with a GitHub client authenticated via PAT.
        """
        super().__init__()
        try:
            # Authenticate with the provided Personal Access Token
            auth = Github(github_pat)
            # Verify authentication by getting the user's login
            auth.get_user().login
            self.github_client = auth
            logging.info("GitHubPRTool: Authenticated with GitHub successfully.")
        except Exception as e:
            logging.error(f"GitHubPRTool: Failed to authenticate with GitHub PAT. {e}")
            # Propagate the error to stop execution if auth fails
            raise Exception(f"Failed to authenticate with GitHub PAT: {e}")

    def _run(self, pr_url: str) -> str:
        """
        Reads the content of all .py files from the specified GitHub PR.
        
        Args:
            pr_url: The full URL of the GitHub Pull Request.
        
        Returns:
            A string containing the concatenated content of all .py files,
            or an error message.
        """
        try:
            # Parse the PR URL to get owner, repo, and PR number
            # e.g., "https://github.com/user/repo/pull/1"
            parts = pr_url.strip("/").split("/")
            if len(parts) < 4 or parts[-2] != "pull":
                return f"Error: Invalid PR URL format. Expected '.../owner/repo/pull/number'."
            
            pr_number = int(parts[-1])
            repo_name = f"{parts[-4]}/{parts[-3]}" # Format: "user/repo"
            
            logging.info(f"Tool: Accessing repo '{repo_name}', PR #{pr_number}")

            # Get the repository and pull request objects
            repo = self.github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
            # Get the SHA of the PR's head commit to fetch correct file versions
            head_sha = pr.head.sha
            
            all_code_content = ""
            
            # Iterate through all files associated with the pull request
            files = pr.get_files()
            for file in files:
                # We only care about Python files for this agent
                if file.filename.endswith(".py"):
                    logging.info(f"Tool: Reading file '{file.filename}'")
                    
                    # Get the full content of the file from the PR branch
                    content_obj = repo.get_contents(file.filename, ref=head_sha)
                    file_content = content_obj.decoded_content.decode('utf-8')
                    
                    # Append file content with clear separators
                    all_code_content += f"\n\n--- START OF FILE: {file.filename} ---\n"
                    all_code_content += file_content
                    all_code_content += f"\n--- END OF FILE: {file.filename} ---\n"
            
            if not all_code_content:
                return "No .py files were found in this Pull Request."
                
            return all_code_content

        except Exception as e:
            logging.error(f"Error while processing GitHub PR: {e}")
            return f"Error while processing GitHub PR: {e}"


class PytestExecutionTool(BaseTool):
    """
    A custom tool to securely execute a self-contained pytest script
    in a temporary file using the current Python environment.
    """
    name: str = "Pytest Execution Tool"
    description: str = "Saves and runs a self-contained pytest script in a temporary file and returns the output."

    def _run(self, test_code: str) -> str:
        """
        Runs a self-contained pytest script in a subprocess and captures the output.
        
        Args:
            test_code: A string containing the full pytest script.
        
        Returns:
            A string containing the stdout/stderr from the pytest execution.
        """
        try:
            # Create a temporary file to write the test code into
            with tempfile.NamedTemporaryFile(delete=False, suffix="_test.py", mode='w', encoding='utf-8', dir=".") as test_file:
                test_file.write(test_code)
                test_file_path = test_file.name

            logging.info(f"Tool: Running tests from temporary file {test_file_path}")

            # Get the path to the python executable *inside the current virtual environment*
            python_executable = sys.executable

            # Run pytest in a new subprocess
            process = subprocess.run(
                [python_executable, "-m", "pytest", test_file_path],
                capture_output=True, # Capture stdout and stderr
                text=True,
                timeout=60 # Add a 60-second timeout for safety
            )

            # Clean up the temporary file
            os.unlink(test_file_path)

            # Return the results
            if process.returncode == 0:
                return f"All tests passed!\n\nOutput:\n{process.stdout}"
            else:
                return f"Tests FAILED!\n\nSTDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}"

        except Exception as e:
            # Ensure cleanup happens even if there's an error
            if 'test_file_path' in locals() and os.path.exists(test_file_path):
                os.unlink(test_file_path)
            logging.error(f"Error during test execution: {e}")
            return f"Error during test execution: {e}"

# --- NEW: Tool 3: GitHub PR Comment Tool ---
class GitHubPRCommentTool(BaseTool):
    """
    A custom tool for posting a comment to a GitHub Pull Request.
    This tool authenticates using a GitHub Personal Access Token (PAT).
    """
    name: str = "GitHub PR Comment Tool"
    description: str = "Posts a comment on a specific GitHub Pull Request URL."
    github_client: Github = None

    def __init__(self, github_pat: str):
        """
        Initializes the tool with an authenticated GitHub client.
        """
        super().__init__()
        try:
            # Authenticate with the provided Personal Access Token
            auth = Github(github_pat)
            auth.get_user().login # Verify auth
            self.github_client = auth
            logging.info("GitHubPRCommentTool: Authenticated with GitHub successfully.")
        except Exception as e:
            logging.error(f"GitHubPRCommentTool: Failed to authenticate with GitHub PAT. {e}")
            raise Exception(f"Failed to authenticate with GitHub PAT: {e}")

    def _run(self, pr_url: str, comment: str) -> str:
        """
        Posts a comment to the specified GitHub PR.
        
        Args:
            pr_url: The full URL of the GitHub Pull Request.
            comment: The markdown text to post as a comment.
        
        Returns:
            A string confirming success or reporting an error.
        """
        try:
            # Parse the PR URL
            parts = pr_url.strip("/").split("/")
            if len(parts) < 4 or parts[-2] != "pull":
                return "Error: Invalid PR URL format. Expected '.../owner/repo/pull/number'."
            
            pr_number = int(parts[-1])
            repo_name = f"{parts[-4]}/{parts[-3]}"
            
            logging.info(f"Tool: Posting comment to repo '{repo_name}', PR #{pr_number}")
            
            # Get the repo and PR
            repo = self.github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
            # Post the comment
            pr.create_issue_comment(comment)
            
            return "Comment posted successfully."
        except Exception as e:
            logging.error(f"Error posting comment: {e}")
            return f"Error posting comment: {e}"


# --- 4. Main Application Logic ---

def main():
    """
    Main function to set up and run the AutoDevOps Crew.
    """
    
    # --- Setup ---
    try:
        # Load API keys from .env and get the GitHub PAT
        github_pat = setup_environment()
    except ValueError as e:
        logging.error(e)
        sys.exit(1) # Exit if keys are missing

    # --- 1. Instantiate Tools ---
    try:
        # Pass the GitHub PAT to the tool constructors
        github_read_tool = GitHubPRTool(github_pat=github_pat)
        github_comment_tool = GitHubPRCommentTool(github_pat=github_pat) # Instantiate new tool
    except Exception as e:
        logging.error(f"Failed to initialize GitHub tools: {e}")
        sys.exit(1) # Exit if GitHub auth fails
    
    pytest_tool = PytestExecutionTool()
    logging.info("All tools instantiated successfully.")

    # --- 2. Define Agents ---
    code_reviewer = Agent(
        role='Expert Python Code Reviewer',
        goal='Analyze Python code from a GitHub PR for bugs, style, and performance. Provide corrected code.',
        backstory=(
            "You are a meticulous Senior Software Engineer. You read code from PRs, "
            "identify all issues, and provide complete, corrected code blocks for review."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[github_read_tool],
        llm=LLM_MODEL
    )

    test_generator = Agent(
        role='Software Quality Assurance Engineer',
        goal='Generate comprehensive, self-contained pytest unit tests for given Python code.',
        backstory=(
            "You are a skilled QA Engineer. You write pytest files that are self-contained. "
            "You MUST include all necessary imports (like 'pytest', 'torch', 'flask') "
            "and use '@pytest.fixture' and 'unittest.mock.patch' to mock any external "
            "dependencies (like file access, API calls, or model loading) so the test can run in isolation."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=LLM_MODEL
    )

    test_executor = Agent(
        role='Software Test Executor',
        goal='Run the provided self-contained pytest script and report the results.',
        backstory=(
            "You are an execution bot. You take a full Python script containing pytest tests, "
            "run it using the Pytest Execution Tool, and report the full, unfiltered output."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[pytest_tool],
        llm=LLM_MODEL
    )
    
    # --- NEW: Define the Report Agent ---
    report_agent = Agent(
        role='DevOps Reporter',
        goal='Summarize the code review and test execution results and post them as a comment on the original GitHub Pull Request.',
        backstory=(
            "You are the final step in the CI pipeline. You take the full code review "
            "and the final test results, format them into a single, easy-to-read "
            "markdown comment, and post it to the PR using your tool."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[github_comment_tool], # Give the agent the comment tool
        llm=LLM_MODEL
    )
    logging.info("All agents defined.")

    # --- 3. Define Tasks ---
    review_task = Task(
        description=(
            f"Use the 'GitHub PR File Reader' tool to read all .py files from the PR: {PR_TO_REVIEW}. "
            "Then, review the code. Provide a report *and* the full, corrected code for *each* file."
        ),
        expected_output=(
            "1. A markdown report summarizing findings for each file.\n"
            "2. A SEPARATE Python code block for *each* corrected file, starting with "
            "--- START CORRECTED CODE: [filename.py] --- and ending with "
            "--- END CORRECTED CODE: [filename.py] ---. This is critical for the next agent."
        ),
        agent=code_reviewer,
    )

    test_generation_task = Task(
        description=(
            "You will receive a review report and corrected code blocks. "
            "Create a single, runnable pytest script. This script must be self-contained. "
            "It MUST import 'pytest' and 'unittest.mock.patch'. "
            "It MUST import the corrected code by referencing the filename (e.g., 'import sample'). "
            "It MUST use '@patch' or fixtures to mock *all* external dependencies: "
            "- Mock `torch.load` to avoid loading a real model file. "
            "- Mock `subprocess.run` and `subprocess.Popen` to avoid running real training. "
            "- Mock `Image.open`, `base64.b6decode`, etc. to fake image processing. "
            "- Mock any API calls (like `genai.GenerativeModel`)."
        ),
        expected_output=(
            "A single Python code block starting with ```python and ending with ```. "
            "This block must be a complete, runnable pytest file, including all necessary imports "
            "(pytest, patch, etc.) and all mocks required to run without external files or APIs."
        ),
        agent=test_generator,
        context=[review_task] # Depends on the output of the review task
    )

    test_execution_task = Task(
        description=(
            "You will receive a string containing a self-contained pytest script. "
            "Extract the Python code block (starting with ```python). "
            "Then, use the 'Pytest Execution Tool' to run this test code."
        ),
        expected_output=(
            "The full, raw output from the 'Pytest Execution Tool', showing "
            "whether the tests passed or failed."
        ),
        agent=test_executor,
        context=[test_generation_task] # Depends on the output of the test generation task
    )
    
    # --- NEW: Define the Report Task ---
    report_task = Task(
        description=(
            "You will receive context from all previous tasks: the review, the generated tests, and the test execution results. "
            "Your job is to create a single, comprehensive summary comment in markdown. "
            "The comment MUST include: "
            "1. A short summary of the 'Code Review Report' (from the first agent). "
            "2. The *final* test execution results (e.g., '14 passed in 0.10s'). "
            "Once you have this summary, use the 'GitHub PR Comment Tool' to post it. "
            f"The URL to post to is: {PR_TO_REVIEW}"
        ),
        expected_output=(
            "A confirmation message stating 'Comment posted successfully.' "
            "or an error if posting failed."
        ),
        agent=report_agent,
        context=[review_task, test_generation_task, test_execution_task] # Depends on all previous tasks
    )
    logging.info("All tasks defined.")

    # --- 4. Create and Run the Crew ---
    code_analysis_crew = Crew(
        agents=[code_reviewer, test_generator, test_executor, report_agent], # ADD the new agent
        tasks=[review_task, test_generation_task, test_execution_task, report_task], # ADD the new task
        process=Process.sequential, # Run tasks in the order provided
        verbose=True
    )
    logging.info("Crew created.")
    
    # Kick off the crew's execution
    logging.info("Starting Code Analysis Crew...")
    result = code_analysis_crew.kickoff()

    # Print the final result from the last task
    logging.info("Crew Execution Finished!")
    print("\n" + "="*30)
    print(" Final Result (Reporting Status)")
    print("="*30)
    print(result)


# This standard Python construct ensures that main() is called
# only when the script is executed directly (not when imported).
if __name__ == "__main__":
    main()
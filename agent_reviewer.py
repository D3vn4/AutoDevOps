"""
AutoDevOps AI Agent Crew
------------------------
This script launches a multi-agent crew to automate the Pull Request (PR)
code review, testing, and reporting process. It integrates with GitHub for
reading files and posting the final summary.

This version incorporates specialized agents and fixes to run reliably on 
GitHub Actions using environment secrets.
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
from github import Github, Auth # CRITICAL: Import Auth for modern PAT usage

# --- 2. Configuration and Setup ---

# Configure logging for clear, professional-looking output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# PR_TO_REVIEW is used as a fallback if the GitHub Action environment variable (PR_URL) is missing
PR_TO_REVIEW = "https://github.com/D3vn4/AutoDevOps/pull/1"

# LLM Model Selection: Use PRO for complex reasoning, Flash for fast initial processing.
LLM_FAST = "gemini/gemini-2.5-flash"
LLM_REASONING = "gemini/gemini-2.5-pro" 

def setup_environment():
    """
    Loads API keys from the OS environment (Secrets passed by GitHub Actions).
    Validates the keys and sets them for LiteLLM and Tool constructors.
    """
    load_dotenv() # Load local .env variables for local testing fallback
    
    # CRITICAL FIX: Fetch values directly from OS environment (where GitHub Actions puts secrets)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    github_pat = os.getenv("PAT_COMMENT") or os.getenv("GITHUB_PAT") 

    # --- Validation ---
    if not google_api_key:
        logging.error("GOOGLE_API_KEY not found in OS Environment/Secrets.")
        raise ValueError("GOOGLE_API_KEY not found in OS Environment/Secrets.")
    if not github_pat:
        logging.error("GITHUB_PAT/PAT_COMMENT not found in OS Environment/Secrets.")
        raise ValueError("GITHUB_PAT/PAT_COMMENT not found in OS Environment/Secrets.")
    
    # CRITICAL: Set the variables explicitly into os.environ for tools/LiteLLM to access
    os.environ['GOOGLE_API_KEY'] = google_api_key
    os.environ['GITHUB_PAT'] = github_pat 

    logging.info("API Keys loaded successfully and set in OS environment.")
    return github_pat

# --- 3. Custom Agent Tools ---

class GitHubPRTool(BaseTool):
    """Tool to read file contents from a specific GitHub Pull Request."""
    name: str = "GitHub PR File Reader"
    description: str = "Reads all .py files from a given GitHub Pull Request URL."
    github_client: Github = None

    def __init__(self, github_pat: str):
        super().__init__()
        try:
            # FIX: Use Auth.Token() for modern, non-deprecated PyGithub authentication
            auth = Auth.Token(github_pat)
            self.github_client = Github(auth=auth)
            logging.info("GitHubPRTool: Authenticated successfully.")
        except Exception as e:
            logging.error(f"GitHubPRTool: Failed to authenticate with PAT. {e}")
            raise Exception(f"Failed to authenticate with GitHub PAT: {e}")

    def _run(self, pr_url: str) -> str:
        """Reads the content of all .py files from the specified GitHub PR."""
        try:
            parts = pr_url.strip("/").split("/")
            if len(parts) < 4 or parts[-2] != "pull":
                return f"Error: Invalid PR URL format. Expected '.../owner/repo/pull/number'."
            
            pr_number = int(parts[-1])
            repo_name = f"{parts[-4]}/{parts[-3]}"
            
            logging.info(f"Tool: Accessing repo '{repo_name}', PR #{pr_number}")

            repo = self.github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            head_sha = pr.head.sha
            all_code_content = ""
            
            for file in pr.get_files():
                if file.filename.endswith(".py"):
                    logging.info(f"Tool: Reading file '{file.filename}'")
                    content_obj = repo.get_contents(file.filename, ref=head_sha)
                    file_content = content_obj.decoded_content.decode('utf-8')
                    
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
    """Tool to securely execute a self-contained pytest script."""
    name: str = "Pytest Execution Tool"
    description: str = "Saves and runs a self-contained pytest script in a temporary file and returns the output."

    def _run(self, test_code: str) -> str:
        """Runs a self-contained pytest script in a subprocess."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix="_test.py", mode='w', encoding='utf-8', dir=".") as test_file:
                test_file.write(test_code)
                test_file_path = test_file.name

            logging.info(f"Tool: Running tests from temporary file {test_file_path}")

            python_executable = sys.executable

            process = subprocess.run(
                [python_executable, "-m", "pytest", test_file_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            os.unlink(test_file_path)

            if process.returncode == 0:
                return f"All tests passed!\n\nOutput:\n{process.stdout}"
            else:
                return f"Tests FAILED!\n\nSTDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}"

        except Exception as e:
            if 'test_file_path' in locals() and os.path.exists('test_file_path'):
                os.unlink('test_file_path')
            logging.error(f"Error during test execution: {e}")
            return f"Error during test execution: {e}"


class GitHubPRCommentTool(BaseTool):
    """Tool for posting the final summary comment to a GitHub Pull Request."""
    name: str = "GitHub PR Comment Tool"
    description: str = "Posts a comment on a specific GitHub Pull Request URL."
    github_client: Github = None

    def __init__(self, github_pat: str):
        super().__init__()
        try:
            # FIX: Use Auth.Token() for modern, non-deprecated PyGithub authentication
            auth = Auth.Token(github_pat)
            self.github_client = Github(auth=auth)
            logging.info("GitHubPRCommentTool: Authenticated successfully.")
        except Exception as e:
            logging.error(f"GitHubPRCommentTool: Failed to authenticate with PAT. {e}")
            raise Exception(f"Failed to authenticate with GitHub PAT: {e}")

    def _run(self, pr_url: str, comment: str) -> str:
        """Posts a comment to the specified GitHub PR."""
        try:
            parts = pr_url.strip("/").split("/")
            if len(parts) < 4 or parts[-2] != "pull":
                return "Error: Invalid PR URL format. Expected '.../owner/repo/pull/number'."
            
            pr_number = int(parts[-1])
            repo_name = f"{parts[-4]}/{parts[-3]}"
            
            logging.info(f"Tool: Posting comment to repo '{repo_name}', PR #{pr_number}")
            
            repo = self.github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
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
        github_pat = setup_environment()
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    # Use the PR_URL environment variable from GitHub Actions, or fallback to PR_TO_REVIEW constant
    # The Action should be passing PR_URL as an environment variable
    pr_url = os.getenv("PR_URL", PR_TO_REVIEW) 

    # --- 1. Instantiate Tools ---
    try:
        # Pass the GitHub PAT to the tool constructors
        github_read_tool = GitHubPRTool(github_pat=github_pat)
        github_comment_tool = GitHubPRCommentTool(github_pat=github_pat) 
    except Exception:
        # Exit handled inside tool constructors if Auth fails
        sys.exit(1) 
    
    pytest_tool = PytestExecutionTool()
    logging.info("All tools instantiated successfully.")

    # --- 2. Define Agents ---
    
    # Reviewer uses the faster Flash model for initial code parsing
    code_reviewer = Agent(
        role='Expert Python Code Reviewer',
        goal='Analyze Python code from a GitHub PR for bugs, style, and performance. Provide corrected code.',
        backstory="You are a meticulous Senior Software Engineer. You read code from PRs, identify all issues, and provide complete, corrected code blocks for review.",
        verbose=True,
        allow_delegation=False,
        tools=[github_read_tool],
        llm=LLM_FAST
    )

    # Security Agent: Specialized agent for finding vulnerabilities and preparing reports
    security_agent = Agent(
        role='Python Security Auditor',
        goal='Analyze code for common security vulnerabilities (e.g., subprocess, unsanitized input, insecure configs) and report findings separately.',
        backstory="You are a specialist in DevSecOps. Your job is to perform a deep dive into the code provided by the Reviewer and flag any security risks, reporting a high-level summary to the main crew.",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=LLM_REASONING # CRITICAL FIX: Use Pro model for deep security analysis
    )

    # Generator uses the Pro model for complex reasoning (mocking/test logic)
    test_generator = Agent(
        role='Software Quality Assurance Engineer',
        goal='Generate comprehensive, self-contained pytest unit tests for given corrected Python code.',
        backstory="You are a skilled QA Engineer. You write pytest files that are self-contained. You MUST use '@patch' to mock all external dependencies for isolated execution.",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=LLM_REASONING # CRITICAL FIX: Use Pro model
    )

    # Executor uses the Pro model for reliable parsing of the test output
    test_executor = Agent(
        role='Software Test Executor',
        goal='Run the provided self-contained pytest script and report the raw results.',
        backstory="You are an execution bot. You take a full Python script containing pytest tests, run it using the Pytest Execution Tool, and report the full, unfiltered output.",
        verbose=True,
        allow_delegation=False,
        tools=[pytest_tool],
        llm=LLM_REASONING # CRITICAL FIX: Use Pro model
    )
    
    # Reporter uses the Pro model for summarizing complex context and final output generation
    report_agent = Agent(
        role='DevOps Reporter',
        goal='Summarize the code review, security audit, and test execution results and post them as a comment on the original GitHub Pull Request.',
        backstory="You are the final step in the CI pipeline. You take all reports, format them into a single, easy-to-read markdown comment, and post it to the PR.",
        verbose=True,
        allow_delegation=False,
        tools=[github_comment_tool],
        llm=LLM_REASONING # CRITICAL FIX: Use Pro model
    )
    logging.info("All agents defined.")

    # --- 3. Define Tasks ---
    review_task = Task(
        description=(
            f"Use the 'GitHub PR File Reader' tool to read all .py files from the PR: {pr_url}. "
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
    
    security_audit_task = Task(
        description=(
            "Review the 'Code Review Report' from the previous task, paying special attention to the *original* code blocks. "
            "Search for critical security vulnerabilities, especially in I/O operations, subprocess calls, and external data handling. "
            "Generate a formal summary highlighting any high-risk findings."
        ),
        expected_output=(
            "A security summary titled 'Security Audit Results' listing any HIGH or MEDIUM risk vulnerabilities found. "
            "If no vulnerabilities are found, state 'No major security vulnerabilities found'."
        ),
        agent=security_agent,
        context=[review_task]
    )

    test_generation_task = Task(
        description=(
            "You will receive a review report and corrected code blocks. "
            "Create a single, runnable pytest script. This script must be self-contained. "
            "It MUST import 'pytest' and use '@patch' or fixtures to mock all external dependencies."
        ),
        expected_output=(
            "A single Python code block starting with ```python and ending with ```. "
            "This block must be a complete, runnable pytest file, including all necessary imports "
            "(pytest, patch, etc.) and all mocks required to run without external files or APIs."
        ),
        agent=test_generator,
        context=[review_task]
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
        context=[test_generation_task]
    )
    
    report_task = Task(
        description=(
            "You will receive context from four previous tasks: the review, the security audit, the generated tests, and the test execution results. "
            "Your job is to create a single, comprehensive summary comment in markdown for the GitHub PR. "
            "The comment MUST be professionally formatted and include: "
            "1. A high-level summary of the original Code Review Report (Agent 1). "
            "2. The Security Audit Results (Agent 2). "
            "3. The *final* Pytest Execution Results (Agent 4), specifically the 'XX passed/failed' line. "
            "Once you have this summary, use the 'GitHub PR Comment Tool' to post it. "
            f"The URL to post to is: {pr_url}"
        ),
        expected_output=(
            "A confirmation message stating 'Comment posted successfully.' "
            "or an error if posting failed."
        ),
        agent=report_agent,
        context=[review_task, security_audit_task, test_generation_task, test_execution_task]
    )
    logging.info("All tasks defined.")

    # --- 4. Create and Run the Crew ---
    code_analysis_crew = Crew(
        agents=[code_reviewer, security_agent, test_generator, test_executor, report_agent], # ADDED Security Agent
        tasks=[review_task, security_audit_task, test_generation_task, test_execution_task, report_task], # ADDED Security Audit Task
        process=Process.sequential,
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
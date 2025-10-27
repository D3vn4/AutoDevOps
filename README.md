# AutoDevOps Agent (In Progress)

## Overview

This project aims to build an AI-powered agent capable of automating parts of the DevOps lifecycle, specifically focusing on Continuous Integration and Continuous Delivery (CI/CD). The goal is to reduce manual overhead in tasks like code review, test generation, and potentially deployment decisions.

This project addresses the bottleneck of slow, manual processes in software development pipelines by leveraging Large Language Models (LLMs) and agentic frameworks.

## Current Status (MVP - Phase 1 Complete)

* **Code Review Agent:** A basic agent is implemented using **CrewAI**.
* **Functionality:** The agent can currently:
    * Read Python code from a specified local file (`sample.py` by default).
    * Use the **Google Gemini API** (`gemini-flash-latest`) to analyze the code.
    * Generate a code review report identifying potential bugs, style issues (PEP 8), and areas for improvement.
* **Local Execution:** The agent runs locally via a Python script (`agent_reviewer.py`).

## Tech Stack 🛠️

* **Language:** Python 3.10+
* **AI Framework:** CrewAI
* **LLM:** Google Gemini API (via `crewai[google-genai]`)
* **Environment:** `python-dotenv` for API key management

## Setup & Run (Current Version)

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd AutoDevOps-Agent
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # (You'll need to create requirements.txt)
    # Or manually: pip install crewai crewai-tools python-dotenv "crewai[google-genai]" litellm
    ```
4.  **Set up API Key:**
    * Create a `.env` file in the root directory.
    * Add your Google Gemini API key: `GOOGLE_API_KEY=YOUR_API_KEY_HERE`
5.  **Configure File:**
    * Ensure `sample.py` (or the file you want to review) exists in the root directory.
    * Verify the `file_to_review` variable inside `agent_reviewer.py` points to the correct file.
6.  **Run the agent:**
    ```bash
    python agent_reviewer.py
    ```


*This project is currently under active development.*

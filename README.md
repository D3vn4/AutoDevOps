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


*This project is currently under active development.*

# 🤖 AutoDevOps AI Agent Crew

## Overview

**AutoDevOps AI Agent Crew** is a fully automated, multi-agent DevOps system that performs **end-to-end Pull Request analysis** using AI agents.

The system autonomously:
- Reviews code quality and logic
- Runs linting and security scans
- Generates and executes unit tests with coverage
- Posts a **comprehensive review summary directly on the GitHub Pull Request**

It is designed to run **locally or inside GitHub Actions**, using environment secrets and modern GitHub authentication.

---

## 🚀 What This Project Does

Given a GitHub Pull Request URL, the system:

1. Reads all Python (`.py`) files from the PR
2. Runs static analysis (linting + security)
3. Performs AI-powered code review
4. Generates self-contained pytest tests
5. Executes tests with coverage
6. Posts a professional markdown report as a PR comment

All of this happens **without manual intervention**.

---

## 🧠 Architecture Overview

The system uses a **sequential multi-agent architecture** built with CrewAI.


---

## 🧑‍💻 AI Agents

| Agent | Responsibility |
|------|---------------|
| **Senior Python Developer** | Code review, linting analysis, logic feedback |
| **Python Security Auditor** | Bandit-based security scanning |
| **QA Engineer** | Generates self-contained pytest tests |
| **Test Executor** | Executes tests with coverage |
| **DevOps Reporter** | Posts final PR comment |

All agents use the **Gemini 2.5 Flash model** to stay within free-tier rate limits.

---

## 🛠️ Tools & Technologies

### Core Stack
- **Language:** Python 3.10+
- **Agent Framework:** CrewAI
- **LLM:** Google Gemini (`gemini-2.5-flash`)
- **GitHub API:** PyGithub (modern token auth)

### DevOps Tooling
- **Ruff** – Python linting
- **Bandit** – Security scanning
- **Pytest** – Unit testing
- **pytest-cov** – Coverage reporting

---

## 🔐 Authentication & Secrets

The system relies entirely on environment variables (GitHub Actions compatible).

### Required Environment Variables

```env
GOOGLE_API_KEY=your_google_gemini_api_key
GITHUB_PAT=your_github_personal_access_token

Optional:
```env
PR_URL=https://github.com/owner/repo/pull/number

If PR_URL is not provided, the script falls back to:
```bash
https://github.com/D3vn4/AutoDevOps/pull/1

### Installation
###1️⃣ Clone the Repository
```bash
git clone https://github.com/D3vn4/AutoDevOps.git
cd AutoDevOps

### 2️⃣ Install Dependencies
```bash
pip install crewai python-dotenv PyGithub ruff bandit pytest pytest-cov

### ▶️ Running the System
```bash
python agent_reviewer.py

The crew will:
 - Fetch PR files
 - Analyze code
 - Run tests
 - Print final execution status
 - Post a comment on the PR

### 🧪 Testing Strategy

Tests are AI-generated
Tests are fully self-contained
No imports from project modules
All required classes/functions are copied into the test file
Ensures zero ModuleNotFoundError
Coverage is calculated during execution

### 📝 PR Comment Output
The posted PR comment includes:
### ✅ Code Review & Linting
Ruff findings (file, line number, error code)
High-level logic and readability review

### 🔐 Security Audit
Medium and High severity Bandit issues
File and line-level reporting
Explicit confirmation if no major issues exist

### 🧪 Test Execution & Coverage
Pass/fail summary
Raw pytest output
Final coverage percentage

### ⚙️ GitHub Actions Ready
This script is fully compatible with GitHub Actions:
Uses OS environment secrets
Avoids deprecated GitHub auth methods
Handles CI execution safely
Posts results back to the PR automatically

### 🧠 Key Engineering Highlights
Multi-agent orchestration with strict task dependencies
Defensive handling of CI edge cases
Secure secret management
Production-grade logging
Clean separation of responsibilities
Real DevOps automation, not a demo

README.md
<p align="center"> <h1 align="center">📘 Jupyter Course Assistant</h1> <p align="center"> <b>Agentic RAG inside your Jupyter notebook.</b><br> Ask questions about course PDFs, get LaTeX-rendered answers, and even let the agent review your notebook. </p> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.9–3.12-blue?logo=python" /> <img src="https://img.shields.io/badge/OpenAI-Required-green?logo=openai" /> <img src="https://img.shields.io/badge/LlamaIndex-Agentic%20RAG-orange" /> <img src="https://img.shields.io/badge/Jupyter-Notebook%20Extension-yellow?logo=jupyter" /> </p>
🚀 Overview

This repository contains a Jupyter-native AI Teaching Assistant that supports:

📚 Course Q&A using retrieval over embedded PDFs

📝 Notebook Inspector to analyze and improve your current notebook

🧮 Beautiful LaTeX-rendered mathematical explanations

✨ Simple student-friendly API (%%research_agent, %ask, %ask notebook)

⚡ Persistent course memory built once by the instructor

Powered by LlamaIndex, OpenAI, and a custom IPython magic extension.

📁 Repository Structure

```
.
├── agent_test.ipynb
├── research_agent_magic.py
├── course_materials/
│   └── admm_distr_stats.pdf
├── .env.example
├── requirements.txt
├── .gitignore
└── LICENSE
```

🧩 Installation & Setup
1. Clone the Repository

```
git clone https://github.com/FaiyazAzam/jupyter-course-assistant.git
cd jupyter-course-assistant
```

3. Create and Activate a Virtual Environment

macOS / Linux
```
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell
```
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install Dependencies
```
pip install -r requirements.txt
```

4. Configure Environment Variables
```
cp .env.example .env
```

Edit .env:
```
OPENAI_API_KEY=your_openai_api_key_here
LLAMAPARSE_API_KEY=your_llamaparse_key_here   # optional unless re-parsing PDFs
```

⚠️ An OpenAI API key is required.

5. Launch Jupyter
```
jupyter notebook
```

Open agent_test.ipynb.

🧠 Using the Research Agent
1. Load the Extension
```
%load_ext research_agent_magic
```

3. Initialize the Agent
```
%init_research_agent
```

To explicitly specify which notebook to inspect:
```
%init_research_agent --nb agent_test.ipynb
```

❓ Asking Questions
📚 1. Course Q&A (Default)
```
%%research_agent
```
Explain the ADMM x-update step.

📝 2. Notebook Analysis
```
%%research_agent --tool notebook_inspector
```
Summarize this notebook and recommend improvements.

⚡ 3. Shortcuts
Course Questions
```
%ask What is the intuition behind ADMM?
```

Notebook Questions
```
%ask notebook
```
Which sections need clearer explanation?

🔧 Kernel & Environment Notes

Requires Python 3.9–3.12

Restart kernel after updating .env

If LaTeX doesn’t render:

Restart kernel
```
Run %load_ext research_agent_magic
Run %init_research_agent
```

🛠️ Troubleshooting
🔍 Notebook inspector hallucinating?

Run:
```
%init_research_agent --nb your_notebook.ipynb
```

🧮 LaTeX not rendering?

Use:
```
%%research_agent
```

(not print())

🔑 Missing API keys?

Your .env must contain:

```
OPENAI_API_KEY=sk-...
```

👨‍🏫 Instructor Notes

To add more PDFs:
Place PDFs in course_materials/
Update research_agent_magic.py if necessary
Rebuild the PDF embedding index
Students only need %init_research_agent — no preprocessing required

📄 License

MIT License.

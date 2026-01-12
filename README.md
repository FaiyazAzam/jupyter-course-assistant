README.md
<p align="center"> <h1 align="center">📘 Jupyter Course Assistant</h1> <p align="center"> <b>Agentic RAG inside your Jupyter notebook.</b><br> Ask questions about course PDFs, get LaTeX-rendered answers, and even let the agent review your notebook. </p> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.9–3.12-blue?logo=python" /> <img src="https://img.shields.io/badge/LLM-Perplexity%20Sonar-blue" /> <img src="https://img.shields.io/badge/OpenAI-Embeddings%20(for%20instructors)-green?logo=openai" /> <img src="https://img.shields.io/badge/Jupyter-Notebook%20Extension-yellow?logo=jupyter" /> </p>

# 🚀 Overview

This repository contains a Jupyter-native AI Teaching Assistant that supports:

📚 Course Q&A using retrieval over embedded PDFs

📝 Notebook Inspector to analyze and improve your current notebook

🧮 Beautiful LaTeX-rendered mathematical explanations

✨ Simple student-friendly API (%%research_agent, %ask, %ask notebook)

⚡ Persistent course memory built once by the instructor

Powered by Perplexity, OpenAI, and a custom IPython magic extension.

📁 Repository Structure

```
.
├── agent_test.ipynb                # Example notebook demonstrating the agent
├── perplexity_agent_magic.py       # Jupyter magic extension (Perplexity-powered agent)
├── build_course_memory.py          # Instructor script to build course_index/ from PDFs
├── course_materials/               # All source PDFs for the course
│   └── admm_distr_stats.pdf
├── course_index/                   # Persisted vector index (auto-loaded by the agent)
│   ├── docstore.json
│   ├── index_store.json
│   ├── graph_store.json
│   ├── image_vector_store.json
│   ├── default_vector_store.json
│   └── ...                         # Additional LlamaIndex persistence files
├── .env.example                    # Template environment variables for students
├── .gitignore
├── requirements.txt                # Python dependencies
└── LICENSE

```

# 🧩 Installation & Setup

1. Clone the Repository

```
git clone https://github.com/FaiyazAzam/jupyter-course-assistant.git
cd jupyter-course-assistant
```

2. Create and Activate a Virtual Environment

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
For Students:
PPLX_API_KEY=your_perplexity_api_key_here
PPLX_MODEL=sonar-pro      # or sonar-reasoning

For Instructor to build course memory:
OPENAI_API_KEY=your_openai_api_key_here
```

⚠️ Students only need a Perplexity API key.
⚠️ Only instructors need an OpenAI API key (for building course memory).

5. Launch Jupyter
```
jupyter notebook
```

Open agent_test.ipynb.

# 🧠 Using the Research Agent

1. Load the Extension
```
%load_ext perplexity_agent_magic
```

2. Initialize the Agent
```
%init_research_agent
```

To explicitly specify which notebook to inspect:
```
%init_research_agent --nb agent_test.ipynb
```

# ❓ Asking Questions

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

# 🔧 Kernel & Environment Notes

Requires Python 3.9–3.12

Restart kernel after updating .env

If LaTeX doesn’t render:

Restart kernel
Run:

```
Run %load_ext perplexity_agent_magic
Run %init_research_agent
```

# 🛠️ Troubleshooting

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
PPLX_API_KEY=pplx-...
PPLX_MODEL=sonar-pro      # or sonar-reasoning

OPENAI_API_KEY=sk-...
```

# 🎓 Student Notes

## 🔑 Getting Your Perplexity API Key (FREE with Student Verification)

The Jupyter Course Assistant uses Perplexity Sonar as the LLM.
As a student, you can get 1 year of Perplexity Pro for free, which includes monthly API credits.

Follow these steps:

✅ 1. Create a Perplexity Account

1. Go to: https://www.perplexity.ai

2. Click Sign Up and create an account (Google / Apple / Email).

🎓 2. Verify Your Student Status (Free 1-Year Pro Access)

Perplexity offers free Perplexity Pro for 12 months if you verify using your university email.

Steps:

1. Open: https://www.perplexity.ai/student

2. Enter your official .edu or university email address

3. Click Verify

4. Check your email inbox for confirmation

5. Once verified, your account will automatically upgrade to Perplexity Pro (Student Plan)

This includes:

- Faster Sonar models (Sonar-Pro, Sonar-Reaasoning)

- Monthly API credits (typically ~$5 per month)

- No charge required

🔑 3. Generate Your API Key

Once logged in:

Go to Settings → API (or visit https://www.perplexity.ai/settings/api)

1. Click Create API Key

2. Copy the generated key

3. Add it to your .env file:
```
PPLX_API_KEY=your_perplexity_api_key_here
PPLX_MODEL=sonar-pro     # or sonar-reasoning
```

## 📝 Notes for Students

1. You do not need an OpenAI API key

2. Students never need to modify or rebuild the course_index folder.

3. PDFs and embeddings are handled entirely by the instructor

4. Your Perplexity API key is only used when generating answers

5. The agent will automatically load the persisted index from course_index/ and will NOT re-embed or re-process PDFs.


# 👨‍🏫 Instructor Notes

## 🔑 Creating an OpenAI API Key (for embeddings only)

Students do not need an OpenAI key.
This is only for the instructor when building the PDF memory.

1. Visit: https://platform.openai.com/api-keys

2. Log in

3. Click "Create new secret key"

4. Copy the key into your .env file:

```
OPENAI_API_KEY=sk-...
```
This key is used only once when running build_course_memory.py to generate embeddings.

## 📚 Adding or Updating Course PDFs

1. Place new PDFs inside:
```
course_materials/
```

2. Rebuild the course memory:
```
python build_course_memory.py
```

This regenerates the persistent RAG index inside:
```
course_index/
```

3. Push the updated course memory so students receive the new content:
```
course_materials/
course_index/
build_course_memory.py
```

# 📄 License

MIT License.

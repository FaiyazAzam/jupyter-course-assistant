# research_agent_magic.py — Perplexity + citations + LaTeX support
# ------------------------------------------------------------- 
# Simple, stable, and student-friendly Jupyter-native TA agent

import asyncio
import nbformat
import traceback
import nest_asyncio
import os
import re
import shlex
import json
import urllib.request
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython.display import Markdown, display

# LlamaIndex imports
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.workflow import Context
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai_like import OpenAILike  # 🔁 switched from OpenAI -> OpenAILike

nest_asyncio.apply()
load_dotenv()

# ------------------------------
# Remote course index auto-update (GitHub Releases)
# ------------------------------

# Raw URL to latest.json in your GitHub repo
LATEST_JSON_URL = "https://raw.githubusercontent.com/FaiyazAzam/jupyter-course-assistant/main/latest.json"

# Local cache directory for downloaded indexes
CACHE_ROOT = Path.home() / ".jupyter_course_assistant" / "course_index_cache"
CURRENT_PTR_FILE = CACHE_ROOT / "CURRENT_VERSION.txt"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _ensure_latest_course_index(verbose: bool = True) -> Path:
    """
    Ensures the latest course index is available locally.
    Returns the local persist_dir path (nested course_index/ supported).
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[Index] Checking for updates via: {LATEST_JSON_URL}")

    latest = _fetch_json(LATEST_JSON_URL)
    version = latest.get("version")
    url = latest.get("url")

    if not version or not url:
        raise RuntimeError("[Index] latest.json must contain 'version' and 'url'.")

    local_dir = CACHE_ROOT / version
    current_version = _read_text(CURRENT_PTR_FILE)

    # If already installed and current, return it
    if local_dir.exists() and current_version == version:
        persist_dir = local_dir / "course_index"
        if persist_dir.exists():
            if verbose:
                print(f"[Index] Course index is up-to-date ✅ (version={version})")
            return persist_dir

    # If folder exists but pointer is different, activate it
    if local_dir.exists() and current_version != version:
        _write_text(CURRENT_PTR_FILE, version)
        persist_dir = local_dir / "course_index"
        if persist_dir.exists():
            if verbose:
                print(f"[Index] Found cached index version={version} ✅ (activated)")
            return persist_dir

    # Otherwise download + unzip
    zip_path = CACHE_ROOT / f"{version}.zip"
    if verbose:
        print(f"[Index] Downloading course index version={version} ...")
        print(f"[Index] From: {url}")

    _download_file(url, zip_path)

    # Unzip into local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(local_dir)

    # Nested folder expected: local_dir/course_index/
    persist_dir = local_dir / "course_index"
    if not persist_dir.exists():
        # fallback: if files are at root
        persist_dir = local_dir

    _write_text(CURRENT_PTR_FILE, version)

    if verbose:
        print(f"[Index] Installed course index ✅ (version={version})")
        print(f"[Index] Persist dir: {persist_dir}")

    return persist_dir


_AGENT = None
_CTX = None


# -------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------

def _load_notebook_content(nb_path=None):
    """Load and concatenate the text of the current Jupyter notebook for context."""
    try:
        # Find the first .ipynb in the working directory if not explicitly provided
        if nb_path is None:
            nb_files = list(Path.cwd().glob("*.ipynb"))
            if nb_files:
                nb_path = nb_files[0]
            else:
                print("[NotebookLoader] No .ipynb file found in current directory.")
                return ""

        # Safely read the notebook once
        with open(nb_path, "r", encoding="utf-8", errors="ignore") as f:
            nb = nbformat.read(f, as_version=4)

        # Gather all code and markdown cells
        all_cells = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") in ("markdown", "code"):
                all_cells.append(cell.get("source", ""))

        return "\n\n---\n\n".join(all_cells)

    except Exception as e:
        print(f"[NotebookLoader] Could not read notebook ({nb_path}):", e)
        return ""


def _format_latex_equations(text: str) -> str:
    """
    Post-process the model output so LaTeX-y expressions render correctly.

    Heuristics:
      1) Any [ ... ] block that looks like LaTeX (contains \, ^, or _)
         becomes a display-style block:

             $$ 
             ... 
             $$

      2) Simple 'a = b' expressions become inline math: $a = b$,
         unless they are already inside $...$.
    """

    # 1) [ ... ] blocks -> display math blocks
    def wrap_bracketed(match: re.Match) -> str:
        inner = match.group("inner")
        stripped = inner.strip()

        # Skip non-math brackets like [CITATIONS], [CONTEXT], etc.
        if not any(ch in stripped for ch in ("\\", "^", "_")):
            return match.group(0)

        # Don't touch if already math-ish
        if "$" in stripped or "\\[" in stripped or "\\]" in stripped:
            return match.group(0)

        # Force a proper block:
        # blank lines before and after so it becomes its own paragraph
        return f"\n\n$$\n{stripped}\n$$\n\n"

    text = re.sub(r"\[(?P<inner>[^\[\]]+)\]", wrap_bracketed, text)

    # 2) Simple "a = b" -> inline $a = b$
    def wrap_eq(match: re.Match) -> str:
        expr = match.group(0)
        stripped = expr.strip()
        # If it's already wrapped, leave as-is
        if stripped.startswith("$") and stripped.endswith("$"):
            return expr
        return f"${stripped}$"

    text = re.sub(
        r"(?<!\$)([A-Za-z0-9_+\-*/^=<>\\|∇‖]+)\s*=\s*([A-Za-z0-9_+\-*/^=<>\\|∇‖]+)(?!\$)",
        wrap_eq,
        text,
    )

    return text


def _format_response(result_text: str) -> str:
    """Apply LaTeX and Markdown-friendly formatting to model output."""
    text = result_text.strip()
    text = _format_latex_equations(text)
    # Add spacing for clarity
    return text.replace("\n\n", "\n\n---\n\n")


def _display_markdown_response(result_text: str):
    formatted = _format_response(result_text)
    display(Markdown(formatted))


# -------------------------------------------------------------------
# Agent builder and setup
# -------------------------------------------------------------------
def _build_agent(agent_type, llm, tools, verbose=True):
    if agent_type.lower() == "function":
        agent = FunctionAgent(tools=tools, llm=llm, verbose=verbose)
    else:
        agent = ReActAgent(tools=tools, llm=llm, verbose=verbose)

    agent._llm = llm
    agent.llm = llm
    ctx = Context(agent)
    ctx.llm = llm
    return agent, ctx


def _default_setup_if_missing(ns):
    """
    Automatically set up the LLM, embedding model, and tools:
    - Course QnA (uses course_materials/ or a prebuilt memory)
    - Notebook Inspector (summarizes and recommends edits)
    """
    from llama_index.core import Settings, Document
    import nbformat  # ✅ required

    # --- 1️⃣ Configure global defaults for the LLM (Perplexity Sonar via OpenAI-like) ---
    if ns.get("llm") is None:
        pplx_api_key = os.getenv("PPLX_API_KEY")
        if not pplx_api_key:
            print("[LLM] Missing PPLX_API_KEY in environment. Set it in .env.")
        pplx_model = os.getenv("PPLX_MODEL", "sonar-pro")  # or sonar-reasoning

        # OpenAI-compatible wrapper pointing at Perplexity
        ns["llm"] = OpenAILike(
            model=pplx_model,
            api_key=pplx_api_key,
            api_base="https://api.perplexity.ai",
            is_chat_model=True,
            is_function_calling_model=True,  # ReActAgent uses tool-calls
            temperature=0.7,
        )

    Settings.llm = ns["llm"]

    # ⚠️ Important:
    # We *do not* set Settings.embed_model here anymore.
    # Embeddings are created once in build_course_memory.py by the instructor
    # using OpenAIEmbedding. Students only *query* a prebuilt index.

    # --- 2️⃣ Load persisted course index (auto-download via GitHub Releases) ---
    if ns.get("initial_tools") is None:
        tools = []

        try:
            persist_dir = _ensure_latest_course_index(verbose=True)

            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            index = load_index_from_storage(storage_context)
            qe = index.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")

            tools.append(
                QueryEngineTool(
                    query_engine=qe,
                    metadata=ToolMetadata(
                        name="course_qna",
                        description="Answer questions using the instructor-built course index with citations.",
                    ),
                )
            )
            print("[RAG] Course QnA tool registered ✅ (loaded from persisted index)")
        except Exception as e:
            print(f"[RAG] Failed to load persisted course index: {e}")
            print("[RAG] Tip: Ensure latest.json exists and the Release zip URL is correct.")

        ns["initial_tools"] = tools

    print(
        f"[RAG] tools so far: {len(ns['initial_tools'])}, llm: Perplexity Sonar; "
        f"index cache: {CACHE_ROOT}"
    )

    # --- 3️⃣ Add notebook inspector tool ---
    tools = ns.get("initial_tools", []) or []
    # pick explicit notebook path if provided
    explicit_nb = ns.get("NOTEBOOK_PATH", None)
    notebook_text = _load_notebook_content(nb_path=explicit_nb)

    if notebook_text:
        # choose notebook path for metadata
        if explicit_nb:
            nb_path = Path(explicit_nb)
        else:
            nb_path = next(Path.cwd().glob("*.ipynb"), Path("unknown_notebook.ipynb"))
        print(f"[Notebook] Inspector indexing: {nb_path}")

        try:
            nb = nbformat.read(nb_path, as_version=4)
            num_cells = len(nb.get("cells", []))
        except Exception as e:
            print(f"[Notebook] Could not read notebook metadata: {e}")
            num_cells = 0

        doc = Document(
            text=notebook_text,
            metadata={
                "source": "current_notebook",
                "filename": str(nb_path.name),
                "num_cells": num_cells,
            },
        )

        try:
            nb_index = VectorStoreIndex.from_documents([doc])
            nb_qe = nb_index.as_query_engine(similarity_top_k=5, response_mode="compact")

            tools.append(
                QueryEngineTool(
                    query_engine=nb_qe,
                    metadata=ToolMetadata(
                        name="notebook_inspector",
                        description="Answer questions about the current Jupyter notebook, summarize its content, and recommend edits.",
                    ),
                )
            )
            print("[Notebook] Inspector tool registered ✅")
        except Exception as e:
            print(f"[Notebook] Failed to register inspector: {e}")
    else:
        print("[Notebook] Skipped — could not read notebook content.")

    ns["initial_tools"] = tools
    print(f"[RAG] Total tools registered: {len(tools)}")

    # --- Attach tools to global context for runtime access ---
    global _CTX
    if hasattr(_CTX, "_state_store") and hasattr(_CTX._state_store, "_state"):
        _CTX._state_store._state["tools"] = ns["initial_tools"]
        print(f"[RAG] Synced {len(ns['initial_tools'])} tools into runtime context ✅")


def _ensure_agent(ns, agent_type="react", verbose=True, force_reinit=False):
    global _AGENT, _CTX
    if _AGENT is None or force_reinit:
        _default_setup_if_missing(ns)
        llm = ns.get("llm")
        tools = ns.get("initial_tools")
        _AGENT, _CTX = _build_agent(agent_type, llm, tools, verbose)
    return _AGENT, _CTX


def _run_tool_by_name(ns, tool_name: str, prompt: str):
    """Find a registered tool by its metadata.name and run a query against it."""
    tools = ns.get("initial_tools", []) or []
    tool = next(
        (t for t in tools
         if getattr(t, "metadata", None)
         and getattr(t.metadata, "name", "") == tool_name),
        None,
    )
    if tool is None:
        raise RuntimeError(f"No tool named '{tool_name}' is registered.")
    return tool.query_engine.query(prompt)


# -------------------------------------------------------------------
# IPython magics
# -------------------------------------------------------------------
@magics_class
class ResearchAgentMagics(Magics):
    @line_magic
    def init_research_agent(self, line):
        """Initialize or reinitialize the course TA agent.

        Usage:
        %init_research_agent                           # default (react)
        %init_research_agent function                  # old style
        %init_research_agent --agent react             # explicit agent type
        %init_research_agent --nb path/to/notebook.ipynb
        %init_research_agent --notebook path/to/notebook.ipynb
        """
        # ---- parse flags ----
        line = (line or "").strip()
        args = shlex.split(line)
        agent_type = "react"
        nb_path = None

        i = 0
        while i < len(args):
            tok = args[i]
            if tok == "--agent" and i + 1 < len(args):
                agent_type = args[i + 1].strip().lower()
                i += 2
                continue
            if tok in ("--nb", "--notebook") and i + 1 < len(args):
                nb_path = args[i + 1].strip()
                i += 2
                continue
            # bare agent type for backward-compat (e.g., "react" or "function")
            if tok in ("react", "function"):
                agent_type = tok
            i += 1

        # ---- stash explicit notebook path, if given ----
        if nb_path:
            p = Path(nb_path)
            if not p.is_file():
                return f"[NotebookLoader] File not found: {p}"
            resolved = str(p.resolve())
            self.shell.user_ns["NOTEBOOK_PATH"] = resolved
            print(f"[NotebookLoader] Using notebook: {resolved}")

        _ensure_agent(self.shell.user_ns, agent_type=agent_type, force_reinit=True)
        return f"Research agent initialized as: {agent_type.upper()}"

    @cell_magic
    def research_agent(self, line, cell):
        """
        Run a cell prompt through the agent.
        Example:
        %%research_agent --tool notebook_inspector
        Summarize this notebook and suggest structure fixes.
        """
        try:
            _ensure_agent(self.shell.user_ns)

            # --- routing knobs ---
            # allow: %%research_agent --tool notebook_inspector
            line = (line or "").strip()
            forced_tool = None
            if line.startswith("--tool"):
                parts = line.split()
                if len(parts) >= 2:
                    forced_tool = parts[1].strip()

            # allow: %ask notebook ... (sets current_tool beforehand)
            current_tool = self.shell.user_ns.pop("current_tool", None)

            prompt = cell.strip()
            wants_notebook = ("this notebook" in prompt.lower()
                              or "current notebook" in prompt.lower()
                              or "cells in this" in prompt.lower())

            # --- 1) tool routing takes precedence ---
            selected = forced_tool or current_tool
            if selected:
                result = _run_tool_by_name(self.shell.user_ns, selected, prompt)
                _display_markdown_response(str(result))
                return result

            # --- 2) auto-route notebook questions ---
            if wants_notebook:
                result = _run_tool_by_name(self.shell.user_ns, "notebook_inspector", prompt)
                _display_markdown_response(str(result))
                return result

            # --- 3) default: course_qna context + agent ---
            tools = self.shell.user_ns.get("initial_tools", [])
            course_tool = next(
                (t for t in tools
                 if getattr(t, "metadata", None) and t.metadata.name == "course_qna"),
                None,
            )

            async def run_with_retry(prompt_text):
                enforced_prompt = (
                    "You are a helpful English-speaking teaching assistant. "
                    "Always respond in English and render any math expressions using LaTeX. "
                    "Use '$...$' for inline math and '$$...$$' for block equations.\n\n"
                    + prompt_text
                )
                for _ in range(3):
                    try:
                        return await _AGENT.run(enforced_prompt, ctx=_CTX)
                    except Exception as e:
                        if "rate limit" in str(e).lower():
                            await asyncio.sleep(2)
                            continue
                        raise
                raise RuntimeError("Repeated API failure after retries.")

            if course_tool:
                retrieved = course_tool.query_engine.query(prompt)
                citation_str = "\n\n[CITATIONS]\n"
                if hasattr(retrieved, "source_nodes"):
                    for node in getattr(retrieved, "source_nodes", []):
                        meta = node.metadata or {}
                        fname = meta.get("file_name", "course_material")
                        page = meta.get("page_label", "")
                        citation_str += f"- {fname} {f'(page {page})' if page else ''}\n"

                context_prompt = (
                    "You are a helpful English-speaking teaching assistant. "
                    "Always answer in clear, fluent English, even if the source text is in another language.\n\n"
                    "Write all mathematical equations in proper LaTeX syntax, using '$...$' for inline math and "
                    "'$$...$$' for block equations. If the answer contains any formula or expression, always render it "
                    "using LaTeX.\n\n"
                    "Use only the context below from course materials to answer.\n\n"
                    f"[CONTEXT]\n{retrieved}\n\n[QUESTION]\n{prompt}\n"
                    f"{citation_str}"
                )
                result = asyncio.get_event_loop().run_until_complete(run_with_retry(context_prompt))
            else:
                result = asyncio.get_event_loop().run_until_complete(run_with_retry(prompt))

            _display_markdown_response(str(result))
            return result

        except Exception as e:
            tb = traceback.format_exc()
            display(Markdown(f"**Error:** `{e}`\n\n```\n{tb}\n```"))
            return None

    @cell_magic
    def ask(self, line, cell):
        if "notebook" in line:
            self.shell.user_ns["current_tool"] = "notebook_inspector"
        return self.research_agent(line, cell)

    @line_magic
    def tools(self, line=""):
        tools = self.shell.user_ns.get("initial_tools", []) or []
        names = [t.metadata.name for t in tools if getattr(t, "metadata", None)]
        return f"Registered tools: {', '.join(names) or '(none)'}"


def load_ipython_extension(ipython):
    """Register magics for %load_ext."""
    ipython.register_magics(ResearchAgentMagics)

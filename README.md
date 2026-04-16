#### 1. **social-media-marketing-analyst**
An AI-powered marketing assistant that creates product descriptions for retail websites based on technical specifications. It uses LLM capabilities to transform fact sheets into engaging marketing content.

**LangChain features used**:
- `ChatPromptTemplate` - Define reusable prompt templates
- `ChatOllama` - Integration with Ollama LLM models
- `RunnableGenerator` - Chain execution for streaming outputs

**Use case**: Convert technical product specifications into compelling marketing copy for social media and retail platforms.

#### 2. **code-review-agent**
An intelligent code review agent that analyzes code quality, suggests improvements, and identifies potential issues.

**LangChain features used**:
- Agents with tool calling - Execute code analysis tools
- `ChatPromptTemplate` - Define code review prompts
- Message history - Maintain context across multiple review passes
- LLM reasoning - Advanced code quality analysis

**Use case**: Automated code review and quality assurance for development workflows.

#### 3. **research-paper-analyst**
An agent designed to analyze and summarize research papers, extract key insights, and answer questions about academic content.

**LangChain features used**:
- Document loaders - Load research papers
- Text splitters - Chunk large documents
- Retrieval chains - Semantic search and retrieval
- Question-answering pipeline - Extract insights from papers
- Memory management - Track analysis context

**Use case**: Automate research paper analysis and knowledge extraction.

#### 4. **review-analyst**
A sentiment analysis and review processing agent that analyzes customer/user reviews and extracts actionable insights.

**LangChain features used**:
- `ChatPromptTemplate` - Define sentiment and insight prompts
- Output parsers - Structure LLM responses
- Chain composition - Multi-step analysis pipeline
- Prompt engineering - Optimize extraction accuracy
- Batch processing - Process multiple reviews at scale

**Use case**: Process and analyze customer feedback at scale.

#### 5. **simple-agent**
A foundational agent project demonstrating basic LangChain agent patterns, tool integration, and conversation flow.

**LangChain features used**:
- `create_agent` - Build basic agents
- Tool definitions and binding - Create custom tools
- `ChatPromptTemplate` - Simple prompting
- Message handling - Manage conversation state
- Agent loops - Understand agent reasoning and execution

**Use case**: Learning LangChain basics and agent architecture.

#### 6. **it-support-analyst**
A multilingual IT support ticket analyzer that processes customer support messages, detects language, categorizes issues, and generates responses in the original language. Demonstrates language detection, categorization, and translation capabilities.

**LangChain features used**:
- `ChatOllama` - Integration with Ollama LLM models for processing
- `PromptTemplate` - Structured prompt engineering for consistent outputs
- `JsonOutputParser` - Parse LLM responses into structured JSON format
- `Pydantic` models - Define response schema (orig_msg, orig_lang, category, trans_msg, response, trans_response)
- `chain.map()` - Batch processing of multiple support messages

**Use case**: Automate IT support ticket processing with multilingual support, categorization, and response generation.

---

## Running Projects with `uv`

### Prerequisites
- [uv](https://docs.astral.sh/uv/) - Python package and project manager (fast and modern)
- Python 3.10+

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Navigate to the project directory**:
   ```bash
   cd cb-skill-learning/cb-langchain/<project-name>
   ```

### Running a Project

#### Using `uv run` (recommended - simple one-liner)
```bash
# Run with uv directly
uv run main.py
```

#### Using virtual environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Run the project
python main.py
```

#### Using `uv sync` (for reproducible environments)
```bash
# Sync dependencies and run
uv sync
uv run main.py
```

### Project-Specific Setup

#### social-media-marketing-analyst
```bash
cd cb-skill-learning/cb-langchain/social-media-marketing-analyst

# Install dependencies
uv pip install -r requirements.txt

# Ensure product files exist
# - product_factsheet.txt (contains technical specs)
# - product_description.txt (contains product description)

# Run the analyzer
uv run main.py
```

### Useful `uv` Commands

```bash
# List project dependencies
uv pip list

# Add a new dependency
uv pip install <package-name>

# Update all dependencies
uv pip install --upgrade -r requirements.txt

# Show which dependencies are available
uv pip index

# Clean up cache
uv cache prune
```

---

## Project Structure

```
cb-skill-learning/
├── README.md (this file)
├── .gitignore
└── cb-langchain/
    ├── Docs/
    ├── social-media-marketing-analyst/
    │   ├── main.py
    │   ├── product_factsheet.txt
    │   ├── product_description.txt
    │   ├── pyproject.toml
    │   └── uv.lock
    ├── code-review-agent/
    │   ├── main.py
    │   ├── pyproject.toml
    │   └── uv.lock
    ├── research-paper-analyst/
    │   ├── main.py
    │   ├── pyproject.toml
    │   └── uv.lock
    ├── review-analyst/
    │   ├── main.py
    │   ├── pyproject.toml
    │   └── uv.lock
    ├── simple-agent/
    │   ├── main.py
    │   ├── pyproject.toml
    │   └── uv.lock
    └── it-support-analyst/
        ├── main.py
        ├── pyproject.toml
        └── uv.lock
```
---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Ollama Models](https://ollama.ai/)

---

## License

This project is part of the learning collection.

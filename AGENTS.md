# Project Instructions

- Keep the repo English-only.
- Use uv with Python 3.13.13 for local environment management.
- Keep `.env` untracked; it may contain OpenRouter credentials.
- Use `python-dotenv` for dotenv parsing.
- Preserve `substituted` as the central axis.
- Do not add new failure modes in v0.1.
- Do not add Tavily, Notion API, tool execution, RAG, or Langfuse.
- Do not add LLM-as-judge.
- Use Pydantic AI + OpenRouter only for bad response generation.
- Keep classification and trace emission deterministic.
- Tests must not call external APIs.

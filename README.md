# Fact Checker

Multi-model fact verification tool that queries Claude, GPT-4o, and Gemini simultaneously with real-time web search.

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
export GOOGLE_API_KEY='...'

streamlit run fact_checker_pro.py
```

## Features

- **Parallel queries** to 3 LLMs with native web search
- **Streaming results** — see each model's response as it completes
- **Consensus analysis** — shows agreement level across models
- **Source evidence** with snippets from actual web pages

## How It Works

1. Enter a claim to verify
2. Each model searches the web and evaluates the claim
3. Results stream in as each model completes
4. Consensus summary shows overall agreement

## Models & Web Search

| Model | Web Search |
|-------|------------|
| Claude | `web_search_20250305` tool |
| GPT-4o | `web_search_preview` tool |
| Gemini | Google Search grounding |

## Cost

~$0.01-0.05 per claim when querying all 3 models.
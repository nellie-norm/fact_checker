"""
Fact Checker — Multi-Model Verification Tool
"""

import os
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from datetime import date
import re


# ============================================================================
# Rate Limiting
# ============================================================================

DAILY_LIMIT = 20  # queries per day across all users

@st.cache_resource
def get_rate_limiter():
    """Global counter that persists while app is running"""
    return {"date": date.today().isoformat(), "count": 0}

def check_rate_limit():
    """Check if under daily limit"""
    limiter = get_rate_limiter()
    today = date.today().isoformat()
    
    # Reset if new day
    if limiter["date"] != today:
        limiter["date"] = today
        limiter["count"] = 0
    
    return limiter["count"] < DAILY_LIMIT

def increment_counter():
    """Increment the daily counter"""
    limiter = get_rate_limiter()
    limiter["count"] += 1

def get_remaining():
    """Get remaining queries for today"""
    limiter = get_rate_limiter()
    today = date.today().isoformat()
    if limiter["date"] != today:
        return DAILY_LIMIT
    return max(0, DAILY_LIMIT - limiter["count"])

# ============================================================================
# Page Config & Styling
# ============================================================================

st.set_page_config(
    page_title="Straight Facts",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling to match nellnorman.com (careful not to break icon fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cardo:ital,wght@0,400;0,700;1,400&display=swap');
    
    .stApp { 
        background-color: #F9F6EE; 
    }
    
    /* Typography - but NOT spans (to preserve icons) */
    h1, h2, h3, p, label, .stMarkdown {
        font-family: 'Cardo', Georgia, serif !important;
        color: #4B3621 !important;
    }
    
    /* Text inputs */
    .stTextInput input, .stTextArea textarea {
        background: #FFFDF8 !important;
        border: 1px solid #D4C8B8 !important;
        color: #4B3621 !important;
        font-family: 'Cardo', Georgia, serif !important;
    }
    
    /* Button */
    .stButton > button, .stFormSubmitButton > button {
        background: #4B3621 !important;
        color: #F9F6EE !important;
        border: none !important;
        font-family: 'Cardo', Georgia, serif !important;
    }
    
    .stButton > button *, .stFormSubmitButton > button * {
        color: #F9F6EE !important;
    }
    
    /* Links */
    a { color: #CC5500 !important; }
    
    /* Captions */
    .stCaption p {
        color: #6B5B4B !important;
        font-family: 'Cardo', Georgia, serif !important;
    }
    
    /* Expander text only, not icons */
    .streamlit-expanderHeader p {
        font-family: 'Cardo', Georgia, serif !important;
        color: #4B3621 !important;
    }
    
    /* Alert boxes - softer */
    [data-testid="stAlert"] {
        font-family: 'Cardo', Georgia, serif !important;
    }
    
    /* Hide chrome */
    #MainMenu {display: none;}
    footer {display: none;}
    header {display: none;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class SourceEvidence:
    url: str
    title: str = ""
    snippet: str = ""
    supports: bool = True

@dataclass
class FactCheckResult:
    model: str
    verdict: str
    confidence: str
    reasoning: str
    sources: list = field(default_factory=list)
    caveats: str = ""
    error: Optional[str] = None
    raw_response: str = ""


FACT_CHECK_PROMPT = """You are a fact-checker. THOROUGHLY research this claim using web search.

CLAIM: "{claim}"

INSTRUCTIONS:
1. Search multiple times with different queries
2. Break down the claim and verify each part
3. Find authoritative sources (press releases, academic papers, industry reports)
4. Do NOT give up - keep searching with different terms if needed

Respond in JSON:
{{
    "verdict": "ACCURATE" | "PARTIALLY_ACCURATE" | "INACCURATE" | "UNVERIFIABLE",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "reasoning": "2-3 sentences with SPECIFIC evidence from sources",
    "sources": [
        {{"url": "https://...", "title": "Article Title", "snippet": "THE EXACT SENTENCE FROM THE ARTICLE THAT PROVES OR DISPROVES THE CLAIM", "supports": true}}
    ],
    "caveats": "Nuance, conflicts, or data gaps"
}}

CRITICAL INSTRUCTIONS FOR SNIPPETS:
- The snippet MUST be the specific sentence that contains the KEY FACT or STATISTIC relevant to the claim
- Do NOT use generic introductions like "Company X is a leader in..."
- Do NOT use article descriptions or meta text
- FIND the exact sentence with numbers, dates, or facts that verify or contradict the claim
- Example: If checking "Company X raised $50M" → snippet should be "Company X announced a $50M Series B round on March 2024"
- Example: If checking "Product Y is used by 1000 customers" → snippet should be "Product Y now serves over 1,200 enterprise customers"

Mark supports:true if snippet confirms the claim, false if it contradicts."""


EXTRACT_CLAIMS_PROMPT = """Extract factual claims from this text that can be verified as true or false.

TEXT:
{text}

Return ONLY a JSON array of claims. Extract the 5-15 most substantive claims. Skip opinions.

Example: ["Claim one", "Claim two"]

JSON array:"""


# ============================================================================
# API Query Functions
# ============================================================================

def query_claude(claim: str) -> FactCheckResult:
    """Query Claude with web search"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4096,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": FACT_CHECK_PROMPT.format(claim=claim)}]
        )
        
        full_text = ""
        sources = []
        
        if response.content:
            for block in response.content:
                if hasattr(block, 'text') and block.text:
                    full_text += block.text
                if hasattr(block, 'citations') and block.citations:
                    for citation in block.citations:
                        if hasattr(citation, 'url') and citation.url:
                            sources.append(citation.url)
        
        if hasattr(response, 'citations') and response.citations:
            for citation in response.citations:
                if hasattr(citation, 'url') and citation.url:
                    sources.append(citation.url)
        
        return parse_response("Claude", full_text, sources)
        
    except Exception as e:
        return FactCheckResult(model="Claude", verdict="", confidence="",
                               reasoning="", error=str(e))


def query_gpt(claim: str) -> FactCheckResult:
    """Query GPT with web search"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview"}],
            input=FACT_CHECK_PROMPT.format(claim=claim)
        )
        
        full_text = ""
        sources = []
        
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content in item.content:
                        if hasattr(content, 'text') and content.text:
                            full_text += content.text
                        if hasattr(content, 'annotations') and content.annotations:
                            for ann in content.annotations:
                                if hasattr(ann, 'url') and ann.url:
                                    sources.append(ann.url)
        
        if not full_text and hasattr(response, 'output_text') and response.output_text:
            full_text = response.output_text
        
        if not full_text:
            return query_gpt_fallback(claim, "Empty response")
        
        return parse_response("GPT-4o", full_text, sources)
        
    except Exception as e:
        return query_gpt_fallback(claim, str(e))


def query_gpt_fallback(claim: str, original_error: str) -> FactCheckResult:
    """Fallback to regular GPT"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": FACT_CHECK_PROMPT.format(claim=claim)}],
            max_tokens=2048
        )
        
        result = parse_response("GPT-4o", response.choices[0].message.content, [])
        result.caveats = f"Web search unavailable. {result.caveats}".strip()
        return result
        
    except Exception as e:
        return FactCheckResult(model="GPT-4o", verdict="", confidence="",
                               reasoning="", error=f"{original_error}")


def query_perplexity(claim: str) -> FactCheckResult:
    """Query Perplexity - built for search-grounded answers"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=os.environ.get("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
        
        response = client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": FACT_CHECK_PROMPT.format(claim=claim)}],
            max_tokens=2048
        )
        
        full_text = response.choices[0].message.content
        
        # Perplexity returns citations in the response
        sources = []
        if hasattr(response, 'citations') and response.citations:
            for url in response.citations:
                sources.append(url)
        
        return parse_response("Perplexity", full_text, sources)
        
    except Exception as e:
        return FactCheckResult(model="Perplexity", verdict="", confidence="",
                               reasoning="", error=str(e))




# ============================================================================
# Response Parsing
# ============================================================================

def parse_response(model: str, text: str, extracted_sources: list[str]) -> FactCheckResult:
    """Parse JSON response from any model"""
    try:
        text = text.strip()
        
        # If empty response
        if not text:
            return FactCheckResult(
                model=model,
                verdict="UNVERIFIABLE",
                confidence="LOW",
                reasoning="No response received from model.",
                sources=[SourceEvidence(url=url) for url in extracted_sources if url],
                error=None,
                raw_response=""
            )
        
        # Find JSON object - look for outermost { }
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start == -1 or end <= start:
            # No JSON found - try to use the text as reasoning
            return FactCheckResult(
                model=model,
                verdict="UNVERIFIABLE",
                confidence="LOW",
                reasoning=text[:500],
                sources=[SourceEvidence(url=url) for url in extracted_sources if url],
                error=None,
                raw_response=text
            )
        
        json_str = text[start:end]
        
        # Clean up the JSON string
        json_str = json_str.strip()
        
        # Remove markdown code blocks if present
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        # Fix common JSON issues
        json_str = re.sub(r',\s*}', '}', json_str)  # trailing comma before }
        json_str = re.sub(r',\s*]', ']', json_str)  # trailing comma before ]
        
        # Try to parse
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing unescaped quotes in strings - common LLM issue
            # This is a simple fix that works for many cases
            fixed = re.sub(r'(?<!\\)"(?=\w)', '\\"', json_str)
            try:
                data = json.loads(fixed)
            except:
                # Last resort: try to extract fields manually with regex
                data = {}
                
                verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', json_str)
                if verdict_match:
                    data["verdict"] = verdict_match.group(1)
                
                confidence_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', json_str)
                if confidence_match:
                    data["confidence"] = confidence_match.group(1)
                
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', json_str)
                if reasoning_match:
                    data["reasoning"] = reasoning_match.group(1).replace('\\"', '"')
                
                if not data:
                    raise ValueError("Could not extract any fields")
        
        sources = []
        raw_sources = data.get("sources", [])
        
        if isinstance(raw_sources, list):
            for src in raw_sources:
                if isinstance(src, dict):
                    sources.append(SourceEvidence(
                        url=src.get("url", ""),
                        title=src.get("title", ""),
                        snippet=src.get("snippet", ""),
                        supports=src.get("supports", True)
                    ))
                elif isinstance(src, str):
                    sources.append(SourceEvidence(url=src))
        
        existing_urls = {s.url for s in sources}
        for url in extracted_sources:
            if url and url not in existing_urls:
                sources.append(SourceEvidence(url=url))
        
        # Handle caveats - might be string or list
        caveats_raw = data.get("caveats", "")
        if isinstance(caveats_raw, list):
            caveats_raw = "; ".join(str(c) for c in caveats_raw)
        
        return FactCheckResult(
            model=model,
            verdict=data.get("verdict", "UNKNOWN"),
            confidence=data.get("confidence", "UNKNOWN"),
            reasoning=data.get("reasoning", ""),
            sources=sources,
            caveats=str(caveats_raw) if caveats_raw else "",
            raw_response=text
        )
        
    except Exception as e:
        # Graceful fallback - use text as reasoning instead of showing error
        sources = [SourceEvidence(url=url) for url in extracted_sources if url]
        
        # Try to extract verdict from text if possible
        text_lower = (text or "").lower()
        if "accurate" in text_lower and "inaccurate" not in text_lower:
            verdict = "ACCURATE"
        elif "inaccurate" in text_lower or "false" in text_lower:
            verdict = "INACCURATE"
        elif "partial" in text_lower:
            verdict = "PARTIALLY_ACCURATE"
        else:
            verdict = "UNVERIFIABLE"
        
        return FactCheckResult(
            model=model,
            verdict=verdict,
            confidence="LOW",
            reasoning=text[:500] if text else "Could not parse response.",
            sources=sources,
            error=None,  # Don't show error to user
            raw_response=text
        )


# ============================================================================
# Analysis
# ============================================================================

def analyze_consensus(results: list[FactCheckResult]) -> dict:
    """Analyze agreement across models"""
    valid = [r for r in results if not r.error and r.verdict not in ("", "PARSE_ERROR", "UNKNOWN")]
    
    if not valid:
        return {"consensus": "NO_VALID_RESULTS", "agreement_level": 0,
                "majority_verdict": "UNKNOWN", "verdict_breakdown": {},
                "all_sources": [], "unique_caveats": [], "models_responding": 0}
    
    verdicts = [r.verdict for r in valid]
    counts = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    
    most_common = max(counts, key=counts.get)
    agreement_pct = counts[most_common] / len(valid)
    
    all_sources = []
    seen_urls = set()
    for r in valid:
        for src in r.sources:
            url = src.url if isinstance(src, SourceEvidence) else src
            if url and url not in seen_urls:
                all_sources.append(src)
                seen_urls.add(url)
    
    caveats = []
    for r in valid:
        caveat_text = r.caveats
        # Handle if caveats is a list
        if isinstance(caveat_text, list):
            caveat_text = "; ".join(str(c) for c in caveat_text)
        if caveat_text and str(caveat_text).lower() not in ("none", "n/a", ""):
            caveats.append(f"{r.model}: {caveat_text}")
    
    if agreement_pct == 1.0:
        consensus = "FULL_AGREEMENT"
    elif agreement_pct >= 0.66:
        consensus = "MAJORITY_AGREEMENT"
    else:
        consensus = "DISAGREEMENT"
    
    return {
        "consensus": consensus,
        "majority_verdict": most_common,
        "agreement_level": agreement_pct,
        "verdict_breakdown": counts,
        "all_sources": all_sources,
        "unique_caveats": caveats,
        "models_responding": len(valid)
    }


# ============================================================================
# Document Parsing
# ============================================================================

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file"""
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif filename.endswith('.pdf'):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() for page in reader.pages)
        except:
            return ""
    elif filename.endswith('.docx'):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join(para.text for para in doc.paragraphs)
        except:
            return ""
    return ""


def extract_claims(text: str) -> list[str]:
    """Extract claims using GPT"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": EXTRACT_CLAIMS_PROMPT.format(text=text[:8000])}],
            max_tokens=2048
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])
        
        return json.loads(content)
    except:
        return []


# ============================================================================
# UI Components
# ============================================================================

def render_model_card(placeholder, model_name: str, result: Optional[FactCheckResult] = None, loading: bool = False):
    """Render a compact model result card"""
    
    model_config = {
        "Claude": {"icon": "◆", "color": "#8B5CF6"},      # Purple
        "GPT-4o": {"icon": "●", "color": "#10B981"},       # Green
        "Perplexity": {"icon": "◎", "color": "#20B8CD"}    # Cyan
    }
    config = model_config.get(model_name, {"icon": "○", "color": "#666"})
    
    container = placeholder.container()
    
    with container:
        # Model header
        st.markdown(f"**{config['icon']} {model_name}**")
        
        if loading:
            st.caption("⏳ Searching...")
            
        elif result is None:
            st.caption("*No API key*")
            
        elif result.error:
            st.error(f"{result.error[:100]}")
            
        else:
            # Verdict
            verdict_config = {
                "ACCURATE": ("✓ Accurate", "success"),
                "PARTIALLY_ACCURATE": ("◐ Partial", "warning"),
                "INACCURATE": ("✗ Inaccurate", "error"),
                "UNVERIFIABLE": ("? Unverifiable", "info")
            }
            verdict_text, verdict_type = verdict_config.get(result.verdict, ("?", "info"))
            
            if verdict_type == "success":
                st.success(verdict_text)
            elif verdict_type == "warning":
                st.warning(verdict_text)
            elif verdict_type == "error":
                st.error(verdict_text)
            else:
                st.info(verdict_text)
            
            # Reasoning - full text, sanitized
            reasoning = result.reasoning.replace('`', '').replace('```', '')
            st.caption(f"**{result.confidence}** · {reasoning}")
            
            # Sources - vertically stacked, full titles
            if result.sources:
                st.markdown("")  # Small gap
                for src in result.sources[:4]:
                    if isinstance(src, SourceEvidence) and src.url:
                        title = src.title or "Source"
                        st.markdown(f"↳ [{title}]({src.url})")


def fact_check_claim_streaming(claim: str, placeholders: dict):
    """Fact-check with streaming updates"""
    
    queries = {}
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        queries["Claude"] = query_claude
        render_model_card(placeholders["Claude"], "Claude", loading=True)
    else:
        render_model_card(placeholders["Claude"], "Claude", result=None)
    
    if os.environ.get("OPENAI_API_KEY"):
        queries["GPT-4o"] = query_gpt
        render_model_card(placeholders["GPT-4o"], "GPT-4o", loading=True)
    else:
        render_model_card(placeholders["GPT-4o"], "GPT-4o", result=None)
    
    if os.environ.get("PERPLEXITY_API_KEY"):
        queries["Perplexity"] = query_perplexity
        render_model_card(placeholders["Perplexity"], "Perplexity", loading=True)
    else:
        render_model_card(placeholders["Perplexity"], "Perplexity", result=None)
    
    if not queries:
        return [], {}
    
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_model = {
            executor.submit(func, claim): model
            for model, func in queries.items()
        }
        
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
            except Exception as e:
                result = FactCheckResult(model=model, verdict="", confidence="",
                                         reasoning="", error=str(e))
            
            results.append(result)
            placeholders[model].empty()
            render_model_card(placeholders[model], model, result=result)
    
    results.sort(key=lambda r: r.model)
    analysis = analyze_consensus(results)
    
    return results, analysis


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown("## Straight Facts")
    st.caption("Enter a claim and three AI models will verify it using web search")
    
    # Check API keys
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_perplexity = bool(os.environ.get("PERPLEXITY_API_KEY"))
    
    if not any([has_anthropic, has_openai, has_perplexity]):
        st.error("No API keys configured. Set environment variables to continue.")
        st.code("export ANTHROPIC_API_KEY='...'\nexport OPENAI_API_KEY='...'\nexport PERPLEXITY_API_KEY='...'")
        st.stop()
    
    # Single claim input (no tabs)
    with st.form(key="claim_form"):
        claim = st.text_area(
            "Enter a claim to verify",
            placeholder="Enter a claim to fact-check, e.g., 'The Great Wall of China is visible from space with the naked eye.'",
            height=68,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            verify_btn = st.form_submit_button("Verify", type="primary", use_container_width=True)
    
    # Show remaining queries
    remaining = get_remaining()
    st.caption(f"☁ {remaining} queries remaining today")
    
    if verify_btn and claim:
        # Validate input - check for questions
        claim_stripped = claim.strip()
        
        if claim_stripped.endswith("?"):
            st.warning("This tool evaluates **claims**, not questions. Please rephrase as a statement.")
            st.caption("For example, instead of *'Is the Great Wall visible from space?'* try *'The Great Wall is visible from space.'*")
            st.stop()
        
        if len(claim_stripped) < 10:
            st.warning("Please enter a more complete claim to verify.")
            st.stop()
        
        # Detect question patterns
        question_starters = (
            # Standard question words
            "what ", "who ", "where ", "when ", "why ", "how ",
            "which ", "whose ",
            # Contractions
            "what's ", "who's ", "where's ", "when's ", "why's ", "how's ",
            "what're ", "who're ", "where're ",
            "what'd ", "who'd ", "where'd ", "how'd ",
            "what'll ", "who'll ", "where'll ", "how'll ",
            # Verbs
            "is ", "are ", "was ", "were ",
            "do ", "does ", "did ",
            "can ", "could ", "would ", "should ", "will ",
            "have ", "has ", "had ",
            "isn't ", "aren't ", "wasn't ", "weren't ",
            "don't ", "doesn't ", "didn't ",
            "can't ", "couldn't ", "wouldn't ", "shouldn't ", "won't ",
            "haven't ", "hasn't ", "hadn't ",
            # Phrases
            "is there ", "are there ", "is it ", "are they ",
            "tell me ", "explain ", "describe ",
        )
        if claim_stripped.lower().startswith(question_starters):
            st.warning("This looks like a question. Please rephrase as a **claim** to fact-check.")
            st.caption("For example: *'The Great Wall of China is visible from space with the naked eye.'*")
            st.stop()
        
        # Check rate limit
        if not check_rate_limit():
            st.error("Daily limit reached. This is a free demo — please try again tomorrow!")
            st.stop()
        
        # Increment counter before running
        increment_counter()
        
        st.markdown("---")
        
        # Model columns
        col1, col2, col3 = st.columns(3)
        
        placeholders = {
            "Claude": col1.empty(),
            "GPT-4o": col2.empty(),
            "Perplexity": col3.empty()
        }
        
        results, analysis = fact_check_claim_streaming(claim, placeholders)
        
        # Spacing before consensus
        st.markdown("")
        st.markdown("---")
        
        # Consensus
        if results and analysis.get("models_responding", 0) > 0:
            consensus = analysis["consensus"]
            verdict = analysis["majority_verdict"]
            
            consensus_icons = {
                "FULL_AGREEMENT": "✓",
                "MAJORITY_AGREEMENT": "◐",
                "DISAGREEMENT": "✗"
            }
            verdict_display = verdict.replace("_", " ").title()
            consensus_display = consensus.replace("_", " ").title()
            
            st.markdown(f"**{consensus_icons.get(consensus, '?')} {consensus_display}** · Majority: {verdict_display} ({analysis['agreement_level']:.0%})")
            
            # Evidence synthesis
            if analysis.get("all_sources"):
                with st.expander(f"Evidence ({len(analysis['all_sources'])} sources)"):
                    supporting = [s for s in analysis["all_sources"] if isinstance(s, SourceEvidence) and s.supports]
                    contradicting = [s for s in analysis["all_sources"] if isinstance(s, SourceEvidence) and not s.supports]
                    
                    for src in supporting[:4]:
                        title = src.title or "Source"
                        st.markdown(f"✓ [{title}]({src.url})")
                        if src.snippet:
                            st.caption(f'"{src.snippet}"')
                    
                    for src in contradicting[:4]:
                        title = src.title or "Source"
                        st.markdown(f"✗ [{title}]({src.url})")
                        if src.snippet:
                            st.caption(f'"{src.snippet}"')
            
            # Caveats
            if analysis.get("unique_caveats"):
                with st.expander("Caveats"):
                    for c in analysis["unique_caveats"]:
                        st.caption(f"• {c}")


main()
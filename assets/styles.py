# assets/styles.py

CUSTOM_CSS = """
/* ============================================================
   LearnAssist Academic Studio - SaaS Design System
   Palette: Dark Slate (#0B0F19), Navy/Card (#161E2E), Indigo (#6366F1)
   ============================================================ */

/* Root & Global Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(11, 15, 25, 0.6);
}
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.4);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.8);
}

/* Glassmorphism Cards */
.saas-card {
    background: linear-gradient(135deg, rgba(22, 30, 46, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 24px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    margin-bottom: 16px;
    transition: all 0.2s ease-in-out;
}

.saas-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 12px 36px -8px rgba(99, 102, 241, 0.15);
}

/* Onboarding Hero Banner */
.hero-container {
    background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.18) 0%, rgba(11, 15, 25, 0) 70%),
                linear-gradient(180deg, rgba(22, 30, 46, 0.8) 0%, rgba(11, 15, 25, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #A5B4FC;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 9999px;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    line-height: 1.2;
    background: linear-gradient(135deg, #FFFFFF 20%, #C7D2FE 70%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    max-width: 680px;
    margin: 0 auto 24px auto;
    line-height: 1.6;
}

/* Feature Cards Grid */
.feature-card {
    background: rgba(22, 30, 46, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: transform 0.2s, border-color 0.2s;
}

.feature-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

.feature-icon {
    font-size: 1.75rem;
    margin-bottom: 10px;
    display: inline-block;
}

.feature-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 6px;
}

.feature-desc {
    font-size: 0.875rem;
    color: #94A3B8;
    line-height: 1.5;
}

/* Agent Pills */
.agent-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
    letter-spacing: 0.02em;
}

.pill-summary {
    background: rgba(99, 102, 241, 0.18);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

.pill-notes {
    background: rgba(245, 158, 11, 0.18);
    color: #FCD34D;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.pill-quiz {
    background: rgba(16, 185, 129, 0.18);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.pill-qa {
    background: rgba(56, 189, 248, 0.18);
    color: #7DD3FC;
    border: 1px solid rgba(56, 189, 248, 0.4);
}

.pill-cache {
    background: rgba(6, 182, 212, 0.2);
    color: #67E8F9;
    border: 1px solid rgba(6, 182, 212, 0.5);
    animation: cache-pulse 2s infinite ease-in-out;
}

@keyframes cache-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.4); }
    50% { box-shadow: 0 0 8px 2px rgba(6, 182, 212, 0.3); }
}

/* HUD Metric Card */
.hud-metric-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}

.hud-metric-label {
    font-size: 0.72rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

.hud-metric-val {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 2px;
}

/* Citation Chips */
.citation-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}

.citation-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.75rem;
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #CBD5E1;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Interactive Flashcard Card */
.flashcard-wrapper {
    background: linear-gradient(145deg, #161E2E 0%, #0F172A 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 16px;
    padding: 36px 30px;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.6);
    position: relative;
    margin-bottom: 16px;
}

.flashcard-tag {
    position: absolute;
    top: 14px;
    left: 18px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border-radius: 4px;
}

.flashcard-tag-front {
    background: rgba(99, 102, 241, 0.2);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.flashcard-tag-back {
    background: rgba(16, 185, 129, 0.2);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.flashcard-counter {
    position: absolute;
    top: 14px;
    right: 18px;
    font-size: 0.78rem;
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
}

.flashcard-content {
    font-size: 1.15rem;
    font-weight: 600;
    color: #F8FAFC;
    line-height: 1.6;
    max-width: 90%;
    margin-top: 12px;
}

/* MCQ Interactive Card */
.mcq-card {
    background: rgba(22, 30, 46, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
}

.mcq-question {
    font-size: 1.02rem;
    font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 12px;
}

.mcq-result-correct {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6EE7B7;
    padding: 10px 14px;
    border-radius: 8px;
    margin-top: 10px;
    font-size: 0.88rem;
}

.mcq-result-incorrect {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #FCA5A5;
    padding: 10px 14px;
    border-radius: 8px;
    margin-top: 10px;
    font-size: 0.88rem;
}

/* Custom Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.5);
    padding: 6px 8px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    padding: 0 18px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #94A3B8;
    background-color: transparent;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0.15) 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(99, 102, 241, 0.45) !important;
}

/* Primary Button Styling */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    border-color: #6366F1;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3);
}

/* Streamlit Container Border Wrapper for SaaS Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, rgba(22, 30, 46, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
    transition: all 0.2s ease-in-out;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(99, 102, 241, 0.35) !important;
    box-shadow: 0 12px 36px -8px rgba(99, 102, 241, 0.15) !important;
}
"""


def inject_custom_css():
    """Inject SaaS glassmorphism CSS into Streamlit page."""
    import streamlit as st
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)



def agent_badge_html(agent_name: str) -> str:
    """Return an HTML pill badge for the given agent."""
    name = (agent_name or "").lower()
    if "summary" in name:
        return '<span class="agent-pill pill-summary">📊 Summary Agent</span>'
    elif "notes" in name:
        return '<span class="agent-pill pill-notes">📝 Notes Agent</span>'
    elif "question" in name or "quiz" in name:
        return '<span class="agent-pill pill-quiz">🧠 Question Agent</span>'
    elif "cache" in name:
        return '<span class="agent-pill pill-cache">⚡ Semantic Cache</span>'
    elif "smalltalk" in name:
        return '<span class="agent-pill pill-qa">💬 SmallTalk</span>'
    else:
        return f'<span class="agent-pill pill-qa">🤖 {agent_name}</span>'


def citation_chips_html(citations: list) -> str:
    """Return HTML chips for document citations."""
    if not citations:
        return ""
    chips = "".join(
        f'<span class="citation-chip">📄 {c}</span>' for c in citations[:6]
    )
    return f'<div class="citation-container">{chips}</div>'

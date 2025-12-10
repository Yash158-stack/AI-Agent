from difflib import get_close_matches

from agents.summary_agent import SummaryAgent
from agents.question_agent import QuestionAgent
from agents.notes_agent import NotesAgent
from agents.smalltalk_agent import SmallTalkAgent
from agents.keywords import (
    SUMMARY_KEYS,
    QUESTION_KEYS,
    NOTES_KEYS,
    COMPLIMENT_KEYS,
    SMALLTALK_KEYS
)


def fuzzy_match(query, keywords):
    qwords = (query or "").lower().split()
    for w in qwords:
        if w in keywords:
            return True
        if get_close_matches(w, keywords, n=1, cutoff=0.7):
            return True
    return False


def is_compliment(query):
    q = (query or "").lower()
    return any(k in q for k in COMPLIMENT_KEYS)


def orchestrator(query: str, context: str, button_state: dict = None):
    q = (query or "").strip().lower()

    # Button-triggered actions
    if button_state:
        if button_state.get("summary"):
            return SummaryAgent.run(query, context)
        if button_state.get("questions"):
            return QuestionAgent.run(query, context)
        if button_state.get("notes"):
            return NotesAgent.run(query, context)

    # Compliment response
    if is_compliment(q):
        return {
            "agent": "SmallTalkAgent",
            "output": "Thanks! Glad it helped 😊"
        }

    # Small talk
    if SmallTalkAgent.is_smalltalk(query):
        return SmallTalkAgent.run(query)

    # Keyword-based routing
    if fuzzy_match(q, SUMMARY_KEYS):
        return SummaryAgent.run(query, context)

    if fuzzy_match(q, QUESTION_KEYS):
        return QuestionAgent.run(query, context)

    if fuzzy_match(q, NOTES_KEYS):
        return NotesAgent.run(query, context)

    # Default → QA Agent
    from agents.qa_agent import QAAgent
    return QAAgent.run(query, context)

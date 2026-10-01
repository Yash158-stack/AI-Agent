# agents/orchestrator.py
import re
from difflib import get_close_matches

from agents.intent_agent import IntentAgent
from agents.keywords import (
    COMPLIMENT_KEYS,
    NOTES_KEYS,
    QUESTION_KEYS,
    SUMMARY_KEYS,
)
from agents.notes_agent import NotesAgent
from agents.qa_agent import QAAgent
from agents.question_agent import QuestionAgent
from agents.smalltalk_agent import SmallTalkAgent
from agents.summary_agent import SummaryAgent


def is_compliment(query: str) -> bool:
    q = (query or "").strip().lower().rstrip("!.,")
    if not q:
        return False
    for comp in COMPLIMENT_KEYS:
        if q == comp or q.startswith(comp + " ") or q.endswith(" " + comp):
            return True
    return False


def keyword_match(query: str, keywords: list) -> bool:
    q = (query or "").lower()

    # 1. Exact phrase or whole-word match
    for key in keywords:
        if " " in key:
            if key in q:
                return True
        else:
            if re.search(rf"\b{re.escape(key)}\b", q):
                return True

    # 2. Strict fuzzy match on individual words (length >= 4, cutoff=0.85)
    words = re.findall(r"\b\w+\b", q)
    single_key_targets = [k for k in keywords if " " not in k and len(k) >= 4]
    for word in words:
        if len(word) >= 4:
            matches = get_close_matches(word, single_key_targets, n=1, cutoff=0.85)
            if matches:
                return True
    return False


def run_agent_for_intent(intent, query, context):
    if intent == "summary":
        return SummaryAgent.run(query, context)
    if intent == "questions":
        return QuestionAgent.run(query, context)
    if intent == "notes":
        return NotesAgent.run(query, context)
    return QAAgent.run(query, context)


def orchestrator(query: str, context: str, button_state: dict = None):
    q = (query or "").strip()

    # Explicit button-triggered actions have highest priority
    if button_state:
        if button_state.get("summary"):
            return SummaryAgent.run(query, context)
        if button_state.get("questions"):
            return QuestionAgent.run(query, context)
        if button_state.get("notes"):
            return NotesAgent.run(query, context)

    # Compliments
    if is_compliment(q):
        return {
            "agent": "SmallTalkAgent",
            "output": "Thanks! Glad it helped.",
        }

    # Small talk greetings/chitchat (checked with word boundaries)
    if SmallTalkAgent.is_smalltalk(q):
        return SmallTalkAgent.run(query)

    # Fast keyword routing
    if keyword_match(q, SUMMARY_KEYS):
        return SummaryAgent.run(query, context)
    if keyword_match(q, QUESTION_KEYS):
        return QuestionAgent.run(query, context)
    if keyword_match(q, NOTES_KEYS):
        return NotesAgent.run(query, context)

    # Fallback to LLM intent classification
    intent = IntentAgent.classify(query)
    return run_agent_for_intent(intent, query, context)

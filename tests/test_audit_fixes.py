# tests/test_audit_fixes.py
import sys
import os
from unittest.mock import MagicMock

# Force UTF-8 stdout if possible on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Mock google.generativeai if not installed in local environment
try:
    import google.generativeai
except ImportError:
    sys.modules["google.generativeai"] = MagicMock()

try:
    import langchain_huggingface
except ImportError:
    sys.modules["langchain_huggingface"] = MagicMock()

try:
    import langchain_community
    import langchain_community.vectorstores
except ImportError:
    sys.modules["langchain_community"] = MagicMock()
    sys.modules["langchain_community.vectorstores"] = MagicMock()

# Ensure AI-Agent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pickle

from agents.smalltalk_agent import SmallTalkAgent
from agents.orchestrator import is_compliment, keyword_match
from agents.keywords import SUMMARY_KEYS, QUESTION_KEYS, NOTES_KEYS
from cache import _deserialize_embedding, should_cache_response
from chat_engine import _format_chat_history


def test_smalltalk_word_boundary():
    print("Testing SmallTalk word boundary matching...")
    # These contain 'hi' as substring and previously caused false positive routing
    false_positives = [
        "which machine learning algorithm is this?",
        "what is the history of computing?",
        "explain this hierarchy",
        "describe the ship architecture",
        "i think this is correct",
    ]
    for q in false_positives:
        assert not SmallTalkAgent.is_smalltalk(q), f"Failed: '{q}' was falsely identified as smalltalk!"

    # Genuine smalltalk queries
    true_positives = [
        "hi",
        "hello",
        "hey there",
        "how are you",
        "can we talk",
    ]
    for q in true_positives:
        assert SmallTalkAgent.is_smalltalk(q), f"Failed: '{q}' should be identified as smalltalk!"
    print("[PASS] SmallTalk word boundary tests PASSED.")


def test_compliment_filter():
    print("Testing compliment filtering...")
    # Questions with words like 'good' or 'great' should NOT be treated as compliments
    academic_queries = [
        "What are the good features of this system?",
        "Explain the Great Depression discussed in chapter 4",
        "Does this method have a nice asymptotic complexity?",
        "Is this a good design pattern?",
    ]
    for q in academic_queries:
        assert not is_compliment(q), f"Failed: Academic question '{q}' was falsely treated as a compliment!"

    # Genuine compliments
    real_compliments = [
        "thanks",
        "thank you",
        "thank you!",
        "great job",
        "well done",
        "good job",
    ]
    for q in real_compliments:
        assert is_compliment(q), f"Failed: Compliment '{q}' should be recognized!"
    print("[PASS] Compliment filter tests PASSED.")


def test_keyword_matching():
    print("Testing keyword matching and fuzzy cutoff...")
    # Negative / unrelated queries that previously collided
    assert not keyword_match("Why is this not working?", NOTES_KEYS), "Failed: 'not' collided with 'notes'!"
    assert not keyword_match("How do I import pandas?", QUESTION_KEYS), "Failed: 'import' collided with 'important'!"
    assert not keyword_match("Explain photosynthesis", NOTES_KEYS), "Failed: 'explain' collided with 'notes'!"

    # Target queries
    assert keyword_match("summarize the document", SUMMARY_KEYS), "Failed: 'summarize' should match SUMMARY_KEYS"
    assert keyword_match("give me important questions", QUESTION_KEYS), "Failed: 'important questions' should match"
    assert keyword_match("create notes for chapter 1", NOTES_KEYS), "Failed: 'create notes' should match NOTES_KEYS"
    print("[PASS] Keyword matching tests PASSED.")


def test_cache_serialization_and_rejection():
    print("Testing cache serialization and error filtering...")
    # 1. Float32 binary buffer
    original_vec = np.random.rand(384).astype(np.float32)
    binary_blob = original_vec.tobytes()
    deserialized = _deserialize_embedding(binary_blob)
    assert np.allclose(original_vec, deserialized), "Failed: Float32 roundtrip deserialization failed!"

    # 2. Legacy pickle fallback
    pickled_blob = pickle.dumps(original_vec.tolist())
    legacy_deserialized = _deserialize_embedding(pickled_blob)
    assert np.allclose(original_vec, legacy_deserialized), "Failed: Legacy pickle fallback failed!"

    # 3. should_cache_response rejects error messages
    assert not should_cache_response("⚠️ QAAgent error: 429 Quota Exceeded"), "Failed: API error should not be cached!"
    assert not should_cache_response("⚠️ SummaryAgent error: Connection Reset"), "Failed: Connection error should not be cached!"
    assert not should_cache_response("Traceback (most recent call last):"), "Failed: Traceback should not be cached!"
    assert not should_cache_response("I couldn't find relevant info in the uploaded documents."), "Failed: Negative response should not be cached!"

    # 4. Valid response should be cached
    assert should_cache_response("Gradient descent is an optimization algorithm that minimizes the loss function."), "Failed: Valid response was rejected!"
    print("[PASS] Cache serialization and error rejection tests PASSED.")


def test_chat_history_formatting():
    print("Testing chat history formatting...")
    history = [
        ("You", "Who was Nikola Tesla?"),
        ("AI (QAAgent)", "Nikola Tesla was a Serbian-American inventor and engineer."),
        ("You", "When was he born?"),
        ("AI (QAAgent)", "He was born on July 10, 1856."),
    ]
    formatted = _format_chat_history(history, max_turns=4)
    assert "User: Who was Nikola Tesla?" in formatted
    assert "Assistant: Nikola Tesla was a Serbian-American inventor and engineer." in formatted
    assert "User: When was he born?" in formatted
    print("[PASS] Chat history formatting tests PASSED.")


if __name__ == "__main__":
    test_smalltalk_word_boundary()
    test_compliment_filter()
    test_keyword_matching()
    test_cache_serialization_and_rejection()
    test_chat_history_formatting()
    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")

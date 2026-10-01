# tests/test_full_coherence.py
import sys
import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch

# Force UTF-8 stdout if possible on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure mocks for missing optional dependencies in test env
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

# Ensure AI-Agent is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ingest import file_fingerprint, compute_document_set_id, text_splitting_recursive, _base_metadata
from cache import get_cached_response, save_response, should_cache_response, _deserialize_embedding
from chat_engine import handle_conversation, _append_sources, _format_chat_history
from agents.orchestrator import orchestrator, is_compliment, keyword_match
from agents.keywords import SUMMARY_KEYS, QUESTION_KEYS, NOTES_KEYS
from agents.smalltalk_agent import SmallTalkAgent
from db import SessionLocal, QueryCache


def test_fingerprinting_coherence():
    print("Testing document fingerprinting and set id...")
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f1, \
         tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f2:
        f1.write("Document 1 sample contents.")
        f1.flush()
        f2.write("Document 2 sample contents.")
        f2.flush()
        p1, p2 = f1.name, f2.name

    try:
        fp1 = file_fingerprint(p1)
        assert "sha256" in fp1 and "size" in fp1 and "name" in fp1

        # Check order independence
        set_id_a = compute_document_set_id([p1, p2])
        set_id_b = compute_document_set_id([p2, p1])
        assert set_id_a == set_id_b, "Document set ID must be deterministic and order-independent!"

        # Check metadata generator
        meta = _base_metadata(p1, set_id_a)
        assert meta["document_set_id"] == set_id_a
        assert meta["source_name"] == os.path.basename(p1)
        print("[PASS] Document fingerprinting is fully coherent.")
    finally:
        os.remove(p1)
        os.remove(p2)


def test_chunking_coherence():
    print("Testing chunking text splitting...")
    sample_text = "This is a sentence. " * 100
    chunks = text_splitting_recursive(sample_text)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    print("[PASS] Text chunking is fully coherent.")


def test_cache_and_document_scoping():
    print("Testing cache scoping and cross-document isolation...")
    doc_set_1 = "doc_set_alpha_12345"
    doc_set_2 = "doc_set_beta_67890"
    query = "What is the primary methodology?"
    response_1 = "The methodology used is randomized controlled trial."

    # Mock embeddings to produce stable deterministic vector
    mock_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    with patch("cache.get_embeddings_model") as mock_embed_model:
        instance = MagicMock()
        instance.embed_query.return_value = mock_vec.tolist()
        mock_embed_model.return_value = instance

        # 1. Save response for doc_set_1
        save_response(query, response_1, document_set_id=doc_set_1, prompt_version="v1", model_id="gemini-test")

        # 2. Query for doc_set_1 -> Must HIT
        cached_1 = get_cached_response(query, document_set_id=doc_set_1, prompt_version="v1", model_id="gemini-test")
        assert cached_1 == response_1, "Expected cache hit for matching document set!"

        # 3. Query for doc_set_2 -> Must MISS (Zero cross-document leakage)
        cached_2 = get_cached_response(query, document_set_id=doc_set_2, prompt_version="v1", model_id="gemini-test")
        assert cached_2 is None, "Cross-document cache leakage detected! Expected None for doc_set_2."

        # 4. Negative / error response rejection
        save_response("Why failed?", "⚠️ QAAgent error: 500 Internal Server Error", document_set_id=doc_set_1)
        cached_err = get_cached_response("Why failed?", document_set_id=doc_set_1)
        assert cached_err is None, "Error response was incorrectly cached!"

    print("[PASS] Semantic caching and cross-document isolation is 100% coherent.")


def test_orchestrator_routing_coherence():
    print("Testing orchestrator routing and agent dispatch...")
    # 1. Button states take priority
    with patch("agents.summary_agent.SummaryAgent.run") as mock_sum:
        mock_sum.return_value = {"agent": "SummaryAgent", "output": "Summary content"}
        res = orchestrator("any query", "context", button_state={"summary": True})
        assert res["agent"] == "SummaryAgent"

    # 2. Compliments
    res = orchestrator("thanks", "context")
    assert res["agent"] == "SmallTalkAgent"
    assert "Thanks!" in res["output"]

    # 3. Academic questions with 'good' should NOT be compliments
    with patch("agents.qa_agent.QAAgent.run") as mock_qa:
        mock_qa.return_value = {"agent": "QAAgent", "output": "QA answer"}
        res = orchestrator("What is a good algorithm for sorting?", "context")
        # Should NOT be SmallTalkAgent
        assert res["agent"] != "SmallTalkAgent"

    # 4. Smalltalk word boundary
    res = orchestrator("hello", "context")
    assert res["agent"] == "SmallTalkAgent"

    # 'which machine' should NOT trigger SmallTalkAgent
    with patch("agents.qa_agent.QAAgent.run") as mock_qa:
        mock_qa.return_value = {"agent": "QAAgent", "output": "QA answer"}
        res = orchestrator("which machine learning model performs best?", "context")
        assert res["agent"] != "SmallTalkAgent"

    print("[PASS] Orchestrator routing is fully coherent.")


def test_chat_engine_history_and_citations():
    print("Testing chat engine multi-turn history and citations...")
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Einstein published the theory of general relativity in 1915."
    mock_doc.metadata = {"source_name": "relativity.pdf", "chunk_index": 2}
    mock_retriever.invoke.return_value = [mock_doc]

    chat_history = [
        ("You", "Who introduced general relativity?"),
        ("AI (QAAgent)", "Albert Einstein."),
    ]

    with patch("chat_engine.orchestrator") as mock_orch, \
         patch("chat_engine.save_response"), \
         patch("chat_engine.get_cached_response", return_value=None):

        mock_orch.return_value = {
            "agent": "QAAgent",
            "output": "General relativity was published in 1915."
        }

        output, updated_history = handle_conversation(
            user_query="When was it published?",
            retriever=mock_retriever,
            chat_history=chat_history,
            document_set_id="relativity_set_001",
        )

        assert "1915" in output
        assert "Sources used:" in output
        assert "relativity.pdf, chunk 2" in output
        # Verify history grew by 2 items (You + AI)
        assert len(updated_history) == 4
        assert updated_history[-1][0] == "AI (QAAgent)"

    print("[PASS] Chat engine history, citations, and conversation flow are fully coherent.")


if __name__ == "__main__":
    test_fingerprinting_coherence()
    test_chunking_coherence()
    test_cache_and_document_scoping()
    test_orchestrator_routing_coherence()
    test_chat_engine_history_and_citations()
    print("\n🎉 ALL INTEGRATION & COHERENCE TESTS PASSED! 🎉")

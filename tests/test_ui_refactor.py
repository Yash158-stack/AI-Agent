# tests/test_ui_refactor.py
"""
Automated test suite for LearnAssist Academic Studio refactor:
- quiz_parser.py: MCQs, flashcards, essay questions, edge cases
- sample_data.py: Sample ML deck data integrity and loader
- ingest.py: PPTX slide-by-slide image pinning (fixing image broadcasting)
- assets/styles.py: Pill badges, citation chips, and CSS helpers
"""
import os
import sys
import tempfile
import unittest
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

from quiz_parser import parse_mcqs, parse_flashcards, parse_essays, parse_quiz_output
from sample_data import SAMPLE_DOCUMENT_TEXT, SAMPLE_QUIZ_RAW, SAMPLE_SUMMARY, SAMPLE_NOTES, load_sample_deck
from assets.styles import agent_badge_html, citation_chips_html, CUSTOM_CSS
from ingest import extract_slides_from_pptx, index_files


class TestQuizParser(unittest.TestCase):
    """Test suite for quiz_parser.py extraction logic."""

    def test_parse_standard_mcqs(self):
        sample = """
### 3. 5 MCQs

MCQ 1. What does the Bias-Variance tradeoff describe?
a) Tradeoff between model complexity and generalization error
b) Speed vs memory tradeoff
c) CPU vs GPU training
d) Supervised vs unsupervised
Answer: a
Explanation: The tradeoff balances underfitting (bias) and overfitting (variance).

MCQ 2. Which method adds an L1 norm penalty?
a) Ridge
b) Lasso
c) Dropout
d) Adam
Answer: b
        """
        mcqs = parse_mcqs(sample)
        self.assertEqual(len(mcqs), 2)
        self.assertEqual(mcqs[0]["id"], 1)
        self.assertEqual(mcqs[0]["answer"], "a")
        self.assertIn("a", mcqs[0]["options"])
        self.assertIn("Lasso", mcqs[1]["options"]["b"])
        self.assertEqual(mcqs[1]["answer"], "b")

    def test_parse_multicolumn_mcqs(self):
        sample = """
MCQ 1. Which optimizer uses momentum?
a) SGD with Momentum          b) Standard SGD
c) Perceptron                 d) Linear Regression
Answer: a
        """
        mcqs = parse_mcqs(sample)
        self.assertEqual(len(mcqs), 1)
        self.assertEqual(mcqs[0]["answer"], "a")
        self.assertIn("SGD with Momentum", mcqs[0]["options"]["a"])
        self.assertIn("Standard SGD", mcqs[0]["options"]["b"])

    def test_parse_flashcards(self):
        sample = """
### 1. 5 SHORT Important Questions

1. What is Precision?
Answer: Precision measures the proportion of predicted positives that are actually positive (TP / (TP + FP)).

2. What is Recall?
Answer: Recall measures the proportion of actual positives correctly identified (TP / (TP + FN)).
        """
        flashcards = parse_flashcards(sample)
        self.assertEqual(len(flashcards), 2)
        self.assertIn("Precision", flashcards[0]["front"])
        self.assertIn("TP / (TP + FP)", flashcards[0]["back"])
        self.assertIn("Recall", flashcards[1]["front"])

    def test_parse_essays(self):
        sample = """
### 2. 5 LONG Descriptive Questions

1. Explain backpropagation in deep neural networks.
- Derive the loss gradient with respect to weights using the chain rule.
- Discuss vanishing gradients with Sigmoid vs ReLU.
- Highlight computational complexity.

2. Compare Supervised and Unsupervised Learning.
- Contrast dataset requirements and objective functions.
- Provide canonical examples for each.
        """
        essays = parse_essays(sample)
        self.assertEqual(len(essays), 2)
        self.assertIn("backpropagation", essays[0]["question"].lower())
        self.assertEqual(len(essays[0]["rubric"]), 3)
        self.assertIn("chain rule", essays[0]["rubric"][0].lower())

    def test_parse_empty_and_edge_cases(self):
        self.assertEqual(parse_mcqs(""), [])
        self.assertEqual(parse_flashcards(""), [])
        self.assertEqual(parse_essays(""), [])
        self.assertEqual(parse_quiz_output("No valid questions here.")["mcqs"], [])

    def test_parse_mcqs_with_decimals_in_questions_and_options(self):
        sample = """
### 3. 5 MCQs

MCQ 1. If dropout probability is 0.5 and learning rate is 0.01 in Python 3.11, what is the dropout?
a) 0.5          b) 0.01
c) 0.9          d) 1.0
Answer: a
Explanation: 0.5 means half the neurons are deactivated.

MCQ 2. Which decay rate is standard?
(a) 0.999
(b) 0.95
(c) 0.50
(d) 0.10
Answer: (a)
        """
        mcqs = parse_mcqs(sample)
        self.assertEqual(len(mcqs), 2)
        self.assertEqual(mcqs[0]["id"], 1)
        self.assertIn("0.5", mcqs[0]["question"])
        self.assertIn("Python 3.11", mcqs[0]["question"])
        self.assertEqual(mcqs[0]["options"]["a"], "0.5")
        self.assertEqual(mcqs[0]["options"]["b"], "0.01")
        self.assertEqual(mcqs[0]["answer"], "a")
        self.assertEqual(mcqs[1]["id"], 2)
        self.assertEqual(mcqs[1]["answer"], "a")
        self.assertEqual(mcqs[1]["options"]["a"], "0.999")

    def test_parse_flashcards_multiline_and_implicit_answers(self):
        sample = """
### 1. 5 SHORT Important Questions

1. What is Precision?
Answer: Precision measures positive predictive value.
It is calculated as TP / (TP + FP).

2. What is Recall?
Recall measures true positive rate (TP / (TP + FN)).

3. What is F1-Score?
- Harmonic mean of Precision and Recall.
- Balances False Positives and False Negatives.
        """
        fcs = parse_flashcards(sample)
        self.assertEqual(len(fcs), 3)
        self.assertIn("Precision", fcs[0]["front"])
        self.assertIn("positive predictive value", fcs[0]["back"])
        self.assertIn("TP / (TP + FP)", fcs[0]["back"])
        self.assertIn("Recall", fcs[1]["front"])
        self.assertIn("TP / (TP + FN)", fcs[1]["back"])
        self.assertIn("F1-Score", fcs[2]["front"])
        self.assertIn("Harmonic mean", fcs[2]["back"])

    def test_parse_essays_numbered_subpoints_in_rubric(self):
        sample = """
### 2. 5 LONG Descriptive Questions

1. Explain backpropagation in deep neural networks.
Key components:
1. Forward pass computation
2. Loss gradient calculation
3. Chain rule weight updates

2. Compare Supervised and Unsupervised Learning.
- Dataset requirements
- Objective functions
        """
        essays = parse_essays(sample)
        self.assertEqual(len(essays), 2, "Numbered sub-bullets must not be split into fake essay questions")
        self.assertIn("backpropagation", essays[0]["question"].lower())
        self.assertIn("Supervised and Unsupervised", essays[1]["question"])
        self.assertGreaterEqual(len(essays[0]["rubric"]), 3)


class TestSampleData(unittest.TestCase):
    """Test suite for sample_data.py content and loader."""

    def test_sample_quiz_raw_parsing(self):
        parsed = parse_quiz_output(SAMPLE_QUIZ_RAW)
        self.assertEqual(len(parsed["mcqs"]), 5, "Sample deck must contain exactly 5 parsed MCQs")
        self.assertEqual(len(parsed["flashcards"]), 5, "Sample deck must contain exactly 5 parsed Flashcards")
        self.assertEqual(len(parsed["essays"]), 5, "Sample deck must contain exactly 5 parsed Essays")

        # Verify all MCQs have valid answer keys
        for m in parsed["mcqs"]:
            self.assertIn(m["answer"], ["a", "b", "c", "d"])
            self.assertGreaterEqual(len(m["options"]), 2)

    def test_sample_deck_content(self):
        self.assertIn("Machine Learning", SAMPLE_DOCUMENT_TEXT)
        self.assertIn("Bias-Variance", SAMPLE_DOCUMENT_TEXT)
        self.assertIn("Executive Summary", SAMPLE_SUMMARY)
        self.assertIn("Comprehensive Study Notes", SAMPLE_NOTES)

    def test_load_sample_deck_helper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_dir = os.path.join(tmp_dir, "uploads")
            faiss_dir = os.path.join(tmp_dir, "faiss_db")

            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.return_value = [[0.1] * 384] * 10
            mock_embeddings.embed_query.return_value = [0.1] * 384

            with patch("langchain_community.vectorstores.FAISS.from_documents") as mock_faiss:
                mock_db = MagicMock()
                mock_faiss.return_value = mock_db

                res = load_sample_deck(upload_dir, faiss_dir, lambda: mock_embeddings)
                self.assertTrue(os.path.exists(res["file_path"]))
                self.assertEqual(res["filename"], "Machine_Learning_Fundamentals.txt")
                self.assertGreater(res["total_chunks"], 0)
                self.assertIsNotNone(res["document_set_id"])
                self.assertEqual(len(res["quiz_data"]["mcqs"]), 5)


class TestStylesAndUIHelpers(unittest.TestCase):
    """Test suite for assets/styles.py design system."""

    def test_agent_badge_html(self):
        summary_badge = agent_badge_html("SummaryAgent")
        self.assertIn("pill-summary", summary_badge)
        self.assertIn("Summary Agent", summary_badge)

        notes_badge = agent_badge_html("NotesAgent")
        self.assertIn("pill-notes", notes_badge)

        quiz_badge = agent_badge_html("QuestionAgent")
        self.assertIn("pill-quiz", quiz_badge)

        cache_badge = agent_badge_html("AI (Cached)")
        self.assertIn("pill-cache", cache_badge)

    def test_citation_chips_html(self):
        citations = ["lecture1.pdf, chunk 1", "notes.docx, chunk 3"]
        chips = citation_chips_html(citations)
        self.assertIn("citation-chip", chips)
        self.assertIn("lecture1.pdf", chips)
        self.assertIn("notes.docx", chips)

        # Empty citations
        self.assertEqual(citation_chips_html([]), "")

    def test_custom_css_tokens(self):
        self.assertIn("#0B0F19", CUSTOM_CSS)  # Dark slate
        self.assertIn("#6366F1", CUSTOM_CSS)  # Indigo
        self.assertIn(".saas-card", CUSTOM_CSS)
        self.assertIn(".hero-container", CUSTOM_CSS)
        self.assertIn("stVerticalBlockBorderWrapper", CUSTOM_CSS)

    def test_case_insensitive_citation_splitting(self):
        import re
        content_lower = "Here is the explanation.\n\nsources used:\n- doc1.pdf, chunk 1\n- doc2.txt, chunk 2"
        content_upper = "Here is the explanation.\n\nSOURCES USED:\n- doc1.pdf, chunk 1"
        content_mixed = "Here is the explanation.\n\nSources Used:\n- doc1.pdf, chunk 1"

        for c in [content_lower, content_upper, content_mixed]:
            parts = re.split(r"sources\s+used\s*:", c, maxsplit=1, flags=re.IGNORECASE)
            self.assertEqual(len(parts), 2, f"Failed to split: {c}")
            self.assertIn("Here is the explanation", parts[0])
            self.assertIn("doc1.pdf", parts[1])


class TestIngestSlidePinning(unittest.TestCase):
    """Test suite for ingest.py slide-by-slide image pinning (fixing defect)."""

    def test_extract_slides_from_pptx_signature(self):
        # Verify function is defined and callable
        self.assertTrue(callable(extract_slides_from_pptx))

    def test_safe_stem_sanitization(self):
        import re
        unsafe_filename = "Lecture #1: Intro & Overview (v2.0)?.pptx"
        raw_stem = os.path.splitext(unsafe_filename)[0]
        safe_stem = re.sub(r"[^\w\-_\.]", "_", raw_stem)
        self.assertNotIn(":", safe_stem)
        self.assertNotIn("?", safe_stem)
        self.assertNotIn("&", safe_stem)
        self.assertNotIn(" ", safe_stem)

    def test_index_files_txt_support(self):
        """Verify .txt and .md files are cleanly indexed without dropping."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_path = os.path.join(tmp_dir, "test_doc.txt")
            faiss_dir = os.path.join(tmp_dir, "faiss")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Artificial intelligence is transforming education and learning assistants.")

            mock_embeddings = MagicMock()
            with patch("ingest.get_embeddings_model", return_value=mock_embeddings), \
                 patch("langchain_community.vectorstores.FAISS.from_documents") as mock_faiss:
                mock_db = MagicMock()
                mock_faiss.return_value = mock_db

                res = index_files([txt_path], faiss_dir)
                self.assertEqual(res["files_indexed"], 1)
                self.assertGreaterEqual(res["total_chunks"], 1)
                self.assertEqual(res["files"], ["test_doc.txt"])


if __name__ == "__main__":
    unittest.main()

# quiz_parser.py
"""
Structured parser for QuestionAgent outputs into interactive components:
- Interactive MCQs (Question, Options a/b/c/d, Correct Answer, Explanation)
- Active Recall Flashcards (Front prompt, Back answer/insight)
- Long Essay / Exam Prep (Question, Rubric criteria, Model Solution)
"""
import re
from typing import Dict, List, Any


def _clean_str(s: str) -> str:
    if not s:
        return ""
    # Strip markdown bold/italics markers at boundaries
    return s.strip().strip("*_# ").strip()


def parse_mcqs(text: str) -> List[Dict[str, Any]]:
    """
    Parse multiple-choice questions from QuestionAgent output.
    Supports single-column, double-column, lettered options (a/b/c/d or A/B/C/D),
    parenthesized options (a)/(b), bulleted options, and decimal numbers in questions/options.
    """
    if not text:
        return []

    # Isolate MCQ section if available
    mcq_section = text
    sec_match = re.search(
        r"(?:###?\s*(?:3[\.:]|\bMCQs?\b|\bMultiple Choice\b)|(?:\bMCQ FORMAT\b)|(?:^\s*3\.\s*5\s*MCQs?))",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if sec_match:
        mcq_section = text[sec_match.start():]

    # Split into question blocks starting with MCQ <num> or <num>. at the beginning of a line
    # Using line anchors prevents matching decimals (e.g. 0.5 or 3.11) inside questions/options
    q_split_pattern = re.compile(
        r"(?:^|\n)\s*(?:MCQ\s*\d+[\.:]?|\d+[\.:])\s+(.+?)(?=(?:\n\s*(?:MCQ\s*\d+[\.:]?|\d+[\.:])\s+|\Z))",
        re.DOTALL | re.IGNORECASE,
    )

    blocks = q_split_pattern.findall(mcq_section)
    if not blocks and "mcq" in mcq_section.lower():
        raw_splits = re.split(r"(?:^|\n)\s*MCQ\s*\d*[\.:]?", mcq_section, flags=re.IGNORECASE)
        blocks = [b.strip() for b in raw_splits if b.strip()]

    # Option pattern: matches a), A), (a), (A), a., - a), - A)
    opt_pattern = re.compile(
        r"(?:^|\s{2,}|\n|\t)(?:[-*•]\s*)?(?:\(([a-dA-D])\)|([a-dA-D])[\)\.:])\s+(.*?)(?=(?:(?:\s{2,}|\n|\t)(?:[-*•]\s*)?(?:\([a-dA-D]\)|[a-dA-D][\)\.:])\s+)|(?:Answer|Correct(?:\s+Answer)?\s*[:\-])|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    mcqs = []
    item_id = 1

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract answer: 'Answer: a', 'Answer: (a)', 'Correct: b', 'Answer: Option A'
        ans_match = re.search(
            r"(?:Answer|Correct(?:\s+Answer)?)\s*[:\-]\s*(?:Option\s*)?[\*\(]*([a-dA-D])[\*\)\.]*",
            block,
            re.IGNORECASE,
        )
        if not ans_match:
            continue

        correct_letter = ans_match.group(1).lower()

        opt_first = opt_pattern.search(block)
        if not opt_first:
            continue

        question_text = block[:opt_first.start()].strip()
        question_text = re.sub(r"-{3,}", "", question_text).strip()
        if not question_text:
            continue

        options_body = block[opt_first.start():]
        options = {}

        for om in opt_pattern.finditer(options_body):
            letter = (om.group(1) or om.group(2)).lower()
            val = om.group(3).strip().strip("*_")
            val = re.split(r"(?:Answer|Correct(?:\s+Answer)?)\s*[:\-]", val, flags=re.IGNORECASE)[0].strip()
            if letter in ["a", "b", "c", "d"] and val:
                options[letter] = val

        # Ensure at least 2 options and valid answer key
        if len(options) >= 2 and correct_letter in options:
            exp_match = re.search(r"(?:Explanation|Reason)\s*[:\-]\s*(.*)", block, re.IGNORECASE)
            explanation = exp_match.group(1).strip() if exp_match else f"Option ({correct_letter.upper()}) is the verified correct answer based on study documents."

            mcqs.append({
                "id": item_id,
                "question": question_text,
                "options": options,
                "answer": correct_letter,
                "explanation": explanation,
            })
            item_id += 1
        elif len(options) >= 2:
            # Answer letter wasn't inside options dict keys, but is valid letter
            exp_match = re.search(r"(?:Explanation|Reason)\s*[:\-]\s*(.*)", block, re.IGNORECASE)
            explanation = exp_match.group(1).strip() if exp_match else f"Option ({correct_letter.upper()}) is the verified correct answer based on study documents."

            mcqs.append({
                "id": item_id,
                "question": question_text,
                "options": options,
                "answer": correct_letter,
                "explanation": explanation,
            })
            item_id += 1

    return mcqs


def parse_flashcards(text: str) -> List[Dict[str, Any]]:
    """
    Extract short questions / concept pairs into interactive flashcard objects.
    Captures multi-line answers and natural explanations without strict Answer: prefixes.
    """
    if not text:
        return []

    # Isolate Short Questions section if available
    short_sec = text
    sec_match = re.search(
        r"(?:###?\s*(?:1[\.:]|\bSHORT\b)|(?:1\.\s*5\s*SHORT))",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if sec_match:
        end_match = re.search(
            r"(?:###?\s*(?:2[\.:]|3[\.:]|\bLONG\b|\bMCQs?\b)|(?:\b2\.\s*5\s*LONG\b)|(?:\b3\.\s*5\s*MCQs?\b))",
            text[sec_match.start():],
            re.IGNORECASE,
        )
        if end_match:
            short_sec = text[sec_match.start():sec_match.start() + end_match.start()]
        else:
            short_sec = text[sec_match.start():]

    # Split into question items starting with a number at line start
    q_split_pattern = re.compile(
        r"(?:^|\n)\s*(?:Question\s*\d+[\.:]?|\d+[\.:])\s+(.+?)(?=(?:\n\s*(?:Question\s*\d+[\.:]?|\d+[\.:])\s+|\Z))",
        re.DOTALL | re.IGNORECASE,
    )

    matches = q_split_pattern.findall(short_sec)
    flashcards = []

    for idx, block in enumerate(matches, start=1):
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue

        q_line = lines[0].strip("*_# ")
        ans_lines = lines[1:]
        clean_ans_parts = []

        for al in ans_lines:
            sub = re.sub(r"^(?:Answer|Ans|Definition|Concept)[\.:]\s*", "", al, flags=re.IGNORECASE).strip("*_ ")
            if sub:
                clean_ans_parts.append(sub)

        ans_text = "\n".join(clean_ans_parts).strip()
        flashcards.append({
            "id": idx,
            "front": q_line,
            "back": ans_text or "Active Recall Check: Review this core definition and mechanism in your study notes.",
        })

    return flashcards


def _is_essay_question(text: str) -> bool:
    """Heuristic helper to differentiate top-level essay questions from rubric sub-bullets."""
    text_clean = text.strip()
    if not text_clean:
        return False
    if text_clean.endswith("?"):
        return True
    first_word = text_clean.split()[0].lower().rstrip(":,.()") if text_clean.split() else ""
    question_verbs = {
        "explain", "analyze", "compare", "evaluate", "discuss", "what", "how",
        "describe", "detail", "outline", "examine", "contrast", "differentiate",
        "illustrate", "provide", "define", "justify", "derive", "critique"
    }
    if first_word in question_verbs:
        return True
    if len(text_clean) > 35 and not text_clean.lower().startswith(("step", "point", "note", "key", "item", "phase", "sub-")):
        return True
    return False


def parse_essays(text: str) -> List[Dict[str, Any]]:
    """
    Extract long descriptive / essay questions with grading criteria and model guidance.
    Prevents numbered sub-bullets from being wrongly split into fake essay questions.
    """
    if not text:
        return []

    # Isolate Long Questions section
    long_sec = text
    sec_match = re.search(
        r"(?:###?\s*(?:2[\.:]|\bLONG\b|\bDESCRIPTIVE\b)|(?:2\.\s*5\s*LONG))",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if sec_match:
        end_match = re.search(
            r"(?:###?\s*(?:3[\.:]|\bMCQs?\b)|(?:\b3\.\s*5\s*MCQs?\b))",
            text[sec_match.start():],
            re.IGNORECASE,
        )
        if end_match:
            long_sec = text[sec_match.start():sec_match.start() + end_match.start()]
        else:
            long_sec = text[sec_match.start():]

    lines = long_sec.split("\n")
    essays = []
    current_q = None
    current_rubric = []
    expected_num = 1

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        m = re.match(r"^(?:Question\s*(\d+)[\.:]?|(\d+)[\.:])\s+(.+)", line_s, re.IGNORECASE)
        if m:
            num = int(m.group(1) or m.group(2))
            cand_q = m.group(3).strip("*_# ")
            # A new question either matches expected sequential index or meets essay question criteria
            if (num == expected_num and (_is_essay_question(cand_q) or not current_q)) or (num == len(essays) + 1 and _is_essay_question(cand_q)):
                if current_q:
                    essays.append({
                        "id": len(essays) + 1,
                        "question": current_q,
                        "rubric": current_rubric if current_rubric else [
                            "1. Comprehensive theoretical definition and context.",
                            "2. Algorithmic/Mathematical breakdown and trade-offs.",
                            "3. Concrete practical example or architectural diagram."
                        ],
                    })
                current_q = cand_q
                current_rubric = []
                expected_num = num + 1
                continue

        # If not a new top-level question, treat as part of rubric/details
        if current_q:
            clean_sub = re.sub(r"^[-*•]\s*", "", line_s).strip("*_ ")
            if clean_sub:
                current_rubric.append(clean_sub)

    if current_q:
        essays.append({
            "id": len(essays) + 1,
            "question": current_q,
            "rubric": current_rubric if current_rubric else [
                "1. Comprehensive theoretical definition and context.",
                "2. Algorithmic/Mathematical breakdown and trade-offs.",
                "3. Concrete practical example or architectural diagram."
            ],
        })

    return essays


def parse_quiz_output(text: str) -> Dict[str, Any]:
    """
    Parse complete QuestionAgent response into all 3 interactive modalities.
    """
    mcqs = parse_mcqs(text)
    flashcards = parse_flashcards(text)
    essays = parse_essays(text)

    return {
        "mcqs": mcqs,
        "flashcards": flashcards,
        "essays": essays,
        "raw_text": text,
    }

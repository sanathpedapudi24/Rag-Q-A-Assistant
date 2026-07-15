"""
evaluate.py

Measures retrieval quality (Recall@k, MRR) of the RAG pipeline against a
labeled set of question -> relevant-source pairs, and optionally measures
end-to-end answer accuracy via simple keyword matching.

Usage:
    python -m evaluation.evaluate --docs data/sample --eval evaluation/eval_qa.json
    python -m evaluation.evaluate --docs data/sample --eval evaluation/eval_qa.json --check-answers
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import RAGPipeline  # noqa: E402


def load_eval_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def reciprocal_rank(results, relevant_source: str) -> float:
    for rank, (_, chunk) in enumerate(results, start=1):
        if chunk.source == relevant_source:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(pipeline: RAGPipeline, eval_set: list[dict], k_values=(1, 3, 5)):
    max_k = max(k_values)
    recall_hits = {k: 0 for k in k_values}
    mrr_total = 0.0

    per_question = []
    for item in eval_set:
        question = item["question"]
        relevant_source = item["relevant_source"]
        results = pipeline.retrieve(question, k=max_k)
        retrieved_sources = [chunk.source for _, chunk in results]

        for k in k_values:
            if relevant_source in retrieved_sources[:k]:
                recall_hits[k] += 1

        rr = reciprocal_rank(results, relevant_source)
        mrr_total += rr

        per_question.append(
            {
                "question": question,
                "relevant_source": relevant_source,
                "retrieved_sources": retrieved_sources,
                "reciprocal_rank": rr,
            }
        )

    n = len(eval_set)
    summary = {f"recall@{k}": recall_hits[k] / n for k in k_values}
    summary["mrr"] = mrr_total / n
    summary["n_questions"] = n
    return summary, per_question


def evaluate_answers(pipeline: RAGPipeline, eval_set: list[dict]):
    correct = 0
    details = []
    for item in eval_set:
        result = pipeline.answer(item["question"])
        answer_lower = result["answer"].lower()
        expected = item.get("expected_answer_contains", [])
        hit = any(kw.lower() in answer_lower for kw in expected) if expected else None
        if hit:
            correct += 1
        details.append({"question": item["question"], "answer": result["answer"], "hit": hit})
    accuracy = correct / len(eval_set) if eval_set else 0.0
    return accuracy, details


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval / answer accuracy")
    parser.add_argument("--docs", default="data/sample", help="Folder of documents to ingest")
    parser.add_argument("--eval", default="evaluation/eval_qa.json", help="Path to eval_qa.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--check-answers",
        action="store_true",
        help="Also call the LLM to check end-to-end answer accuracy (requires GROQ_API_KEY)",
    )
    args = parser.parse_args()

    doc_paths = []
    for ext in ("*.txt", "*.pdf", "*.docx", "*.md"):
        doc_paths.extend(glob.glob(os.path.join(args.docs, ext)))

    if not doc_paths:
        print(f"No documents found in {args.docs}")
        return

    pipeline = RAGPipeline(top_k=args.top_k)
    n_chunks = pipeline.ingest(doc_paths)
    print(f"Ingested {len(doc_paths)} documents -> {n_chunks} chunks\n")

    eval_set = load_eval_set(args.eval)

    summary, per_question = evaluate_retrieval(pipeline, eval_set)
    print("Retrieval accuracy")
    print("-------------------")
    for k, v in summary.items():
        if k == "n_questions":
            continue
        print(f"  {k}: {v:.1%}" if "recall" in k else f"  {k}: {v:.3f}")
    print(f"  n_questions: {summary['n_questions']}\n")

    for q in per_question:
        status = "✓" if q["reciprocal_rank"] > 0 else "✗"
        print(f"  [{status}] {q['question']}")

    if args.check_answers:
        print("\nEnd-to-end answer accuracy (keyword match against the LLM's answer)")
        print("---------------------------------------------------------------------")
        accuracy, details = evaluate_answers(pipeline, eval_set)
        print(f"  answer accuracy: {accuracy:.1%}\n")
        for d in details:
            status = "✓" if d["hit"] else "✗"
            print(f"  [{status}] {d['question']}")


if __name__ == "__main__":
    main()

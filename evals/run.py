"""Run explicit fixture replay or live Gemini checks; never invent scores."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.engine import ResearchEngine
from backend.gemini import GeminiModel
from backend.models import Question
from backend.store import DocumentStore

DATASET = Path(__file__).with_name("cases.json")


class ReplayModel:
    """Replay scripted fixture outputs to check pipeline contracts, not quality."""

    def __init__(self, case: dict[str, Any]) -> None:
        """Retain the fixture's declared output without consulting expectations."""
        self.case = case

    def count_tokens(self, prompt: str) -> int:
        """Use UTF-8 bytes as deterministic test tokens, not provider counts."""
        return len(prompt.encode("utf-8"))

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        """Bind a scripted quote to the first source chunk supplied by retrieval."""
        if "replay_raw" in self.case:
            return self.case["replay_raw"]
        data = json.loads(prompt.split("UNTRUSTED DATA (JSON):\n", 1)[1])
        quote = self.case.get("replay_quote")
        citations = [{"chunk_id": data["documents"][0]["chunk_id"], "quote": quote}] if quote else []
        return json.dumps({"citations": citations})


def evaluate(mode: str = "replay", dataset: Path = DATASET) -> dict[str, Any]:
    """Return measured fixture outcomes with dataset and model provenance."""
    if mode not in {"replay", "live"}:
        raise ValueError("Mode must be replay or live")
    content = dataset.read_bytes()
    cases = json.loads(content)
    results: list[dict[str, Any]] = []
    live_model = GeminiModel() if mode == "live" else None
    for case in cases:
        if mode == "live" and case.get("replay_only"):
            continue
        store = DocumentStore()
        ids = [store.upload(f"source-{index}.txt", text.encode("utf-8"))["document_id"]
               for index, text in enumerate(case["documents"])]
        selected_ids = [ids[index] for index in case.get("scope", range(len(ids)))]
        model = live_model if live_model is not None else ReplayModel(case)
        answer = ResearchEngine(store, model).answer(Question(question=case["question"], document_ids=selected_ids))
        passed = answer.abstained == case["expected_abstained"]
        if case.get("expected_quote"):
            passed = passed and any(item.quote == case["expected_quote"] for item in answer.citations)
        # Infrastructure failures never count as successful abstention in live checks.
        if mode == "live" and answer.reason in {"model_unavailable", "invalid_output_or_context",
                                               "context_exhausted", "invalid_output", "invalid_citation"}:
            passed = False
        results.append({"id": case["id"], "passed": passed, "abstained": answer.abstained,
                        "reason": answer.reason, "citation_count": len(answer.citations)})
    return {"mode": mode, "purpose": "pipeline_contracts" if mode == "replay" else "live_fixture_checks",
            "dataset_sha256": hashlib.sha256(content).hexdigest(),
            "model": live_model.model if live_model else "scripted_fixture_replay",
            "context_tokens": 8192, "reserved_output_tokens": 1024,
            "cases": results, "passed": sum(row["passed"] for row in results), "total": len(results)}


def main() -> None:
    """Print a JSON report and exit nonzero if any executed fixture failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    args = parser.parse_args()
    report = evaluate(args.mode)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()

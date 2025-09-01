"""
DSPy + Gemini 2.5 Flash: Paper Info Extractor
------------------------------------------------
Goal: reliably extract key research artifacts from papers/tech blogs:
  - ablations
  - key deltas vs prior work / literature
  - new findings / contributions
  - training configs (optimizer, LR, batch size, steps/epochs, schedule, seed, dataset, hardware)
  - params (model size) and training/inference cost (GPU hours, $, CO2 if given)

Features
  - Uses DSPy (declarative, self-improving) with Gemini 2.5 Flash via LiteLLM adapter.
  - Structured JSON outputs with a strict schema + validation.
  - Optional RAG over local corpus of PDFs/URLs (quick BM25, pluggable).
  - Self-optimizes with BootstrapFewShot + RandomSearch using your labeled examples.
  - Verifier pass that grounds each extracted claim to quoted evidence spans.

Prereqs
  pip install dspy-ai litellm google-generativeai pypdf rapidfuzz rich
  export GEMINI_API_KEY=...  # Google AI Studio key (direct Gemini API)

If you use Vertex AI instead, also set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON and change MODEL_ID/provider below accordingly.

Quick start
  python dspy_gemini_paper_extractor.py \
    --inputs ./samples  \
    --glob "*.pdf" \
    --out out.jsonl

To optimize on your gold labels
  python dspy_gemini_paper_extractor.py \
    --inputs ./samples \
    --labels labels.jsonl \
    --optimize \
    --out tuned_predictions.jsonl
"""
from __future__ import annotations
import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import dspy
from dspy import Example
from rich import print as rprint

# --- Model wiring (Gemini 2.5 Flash via LiteLLM-backed dspy.LM) ---
# Recommended provider string format with dspy.LM (LiteLLM under the hood):
#   provider/model-id
# For direct Google AI Studio (Gemini API): "gemini/gemini-2.5-flash"
# For Vertex AI: "vertex_ai/gemini-2.5-flash"
MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")
PROVIDER_PREFIX = os.environ.get("DSPY_PROVIDER_PREFIX", "gemini")  # or "vertex_ai"
LM_STRING = f"{PROVIDER_PREFIX}/{MODEL_ID}" if "/" not in MODEL_ID else MODEL_ID

# Optional Gemini "thinking" budget (Gemini 2.5 supports thinking mode)
THINKING_TOKENS = int(os.environ.get("GEMINI_THINKING_TOKENS", "0"))  # 0 disables

# Instantiate LM once
lm = dspy.LM(model=LM_STRING, api_key=os.getenv("GEMINI_API_KEY"),
             cache=False,
             # You can pass provider-specific kwargs via LiteLLM kwargs
             # E.g., structured outputs, thinking, safety settings, etc.
             # See: https://ai.google.dev/gemini-api/docs for latest
             extra_body={"thinking": {"budget_tokens": THINKING_TOKENS}} if THINKING_TOKENS > 0 else None)

dspy.configure(lm=lm)

# --- Utility: cheap PDF/HTML/text loader ---

def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix.lower() in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ")
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

# --- JSON Schema we target ---
SCHEMA = {
    "paper_title": str,
    "paper_venue": str,  # e.g., NeurIPS 2024 (optional if unknown)
    "source": str,       # filepath or URL
    "ablations": list,   # list[Ablation]
    "key_deltas": list,  # list[Delta]
    "new_findings": list, # list[str]
    "training_config": dict, # Dict of config details
    "model_params": dict,    # {"param_count": str/int, "arch": str}
    "costs": dict,           # {"training": {"gpu_hours": float, "$": float}, "inference": {"$": float}, "co2": float}
    "evidence": list         # list[Evidence]
}

# Types (for readability in prompts)
ABLATION_FMT = {
    "name": "<short label, e.g., remove data aug>",
    "setup": "<what changed vs baseline>",
    "metric": "<metric name>",
    "delta": "<signed change vs baseline with unit>"
}

DELTA_FMT = {
    "area": "<topic/benchmark>",
    "baseline_or_prior_work": "<paper/system>",
    "metric": "<metric name>",
    "value": "<this work>",
    "delta": "<improvement/worsening>"
}

TRAINING_CFG_FMT = {
    "dataset": "<name + version/splits>",
    "hardware": "<GPUs / TPUs, count, model>",
    "optimizer": "<AdamW, etc>",
    "batch_size": "<global batch>",
    "lr": "<learning rate schedule>",
    "epochs_or_steps": "<N>",
    "precision": "<fp16/bf16/fp8/etc>",
    "grad_clip": "<value>",
    "seed": "<seed if given>",
    "regularization": "<wd, dropout, label smoothing>",
    "augmentation": "<text/image augs>",
    "checkpoints": "<ckpt path/policy>",
    "code_url": "<repo if present>"
}

# --- DSPy Signatures ---

class ExtractStructured(dspy.Signature):
    """Extract structured research facts from a paper.

    Use ONLY facts present in the text; don't hallucinate. If unknown, set field to null or empty.
    Return strict JSON complying with the SCHEMA, with keys in snake_case.
    """
    paper_text: str = dspy.Input(desc="Full or chunked paper text")
    schema_hint: str = dspy.Input(desc="Mini description of the JSON schema and examples")
    result_json: str = dspy.Output(desc="Valid JSON for the SCHEMA")

class VerifyAndGround(dspy.Signature):
    """Verify each claim is grounded in the text with a direct quote and page/section hint.
    Return a JSON with fields: valid (bool), issues (list[str]), evidence (list[{claim_idx, quote, where}]).
    """
    paper_text: str = dspy.Input()
    extracted_json: str = dspy.Input()
    verification_json: str = dspy.Output()

# --- Modules ---
class Extractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractStructured)

    def forward(self, paper_text: str) -> Dict[str, Any]:
        schema_hint = json.dumps({
            "ablations": [ABLATION_FMT],
            "key_deltas": [DELTA_FMT],
            "new_findings": ["<finding>"] ,
            "training_config": TRAINING_CFG_FMT,
            "model_params": {"param_count": "<e.g., 1.3B>", "arch": "<e.g., ViT-L/14>"},
            "costs": {"training": {"gpu_hours": None, "$": None}, "inference": {"$": None}, "co2": None},
            "evidence": [{"field": "<which field>", "quote": "<verbatim>", "where": "<page/section>"}]
        }, ensure_ascii=False)

        out = self.extract(paper_text=paper_text, schema_hint=schema_hint)
        try:
            data = json.loads(out.result_json)
        except Exception:
            # Try to salvage JSON
            m = re.search(r"\{.*\}", out.result_json, flags=re.S)
            data = json.loads(m.group(0)) if m else {"error": "non-json"}
        return data

class Verifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.verify = dspy.Predict(VerifyAndGround)

    def forward(self, paper_text: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
        out = self.verify(paper_text=paper_text, extracted_json=json.dumps(extracted, ensure_ascii=False))
        try:
            data = json.loads(out.verification_json)
        except Exception:
            m = re.search(r"\{.*\}", out.verification_json, flags=re.S)
            data = json.loads(m.group(0)) if m else {"valid": False, "issues": ["non-json verifier output"]}
        return data

class PaperIE(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = Extractor()
        self.verifier = Verifier()

    def forward(self, paper_text: str) -> Dict[str, Any]:
        extracted = self.extractor(paper_text)
        verification = self.verifier(paper_text, extracted)
        extracted["_verification"] = verification
        return extracted

# --- Simple BM25 retrieval over a local text index (optional) ---
class SimpleBM25:
    def __init__(self, texts: List[str]):
        from rapidfuzz import fuzz
        self.texts = texts
        self.fuzz = fuzz

    def query(self, q: str, k: int = 5) -> List[str]:
        scores = [(self.fuzz.token_set_ratio(q, t), t) for t in self.texts]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scores[:k]]

# --- Optimization (optional) ---

def build_teleprompter(train_examples: List[Example]):
    # You can swap for MIPROv2 / BayesOpt if you like.
    return dspy.BootstrapFewShotWithRandomSearch(
        metric=extraction_metric,
        num_candidates=4,
        num_trials=6,
        teacher_settings={"temperature": 0.2}
    )

# Metric: JSON validity + coverage of key fields

def extraction_metric(gold: Dict[str, Any], pred: Dict[str, Any], trace=None) -> float:
    score = 0.0
    try:
        # minimal field presence
        for key in ["ablations", "key_deltas", "new_findings", "training_config"]:
            if key in pred and pred[key]:
                score += 0.2
        # evidence presence
        if pred.get("evidence"):
            score += 0.2
        # param count or arch
        mp = pred.get("model_params", {})
        if mp.get("param_count") or mp.get("arch"):
            score += 0.1
        # some costs
        if pred.get("costs", {}).get("training"):
            score += 0.1
    except Exception:
        pass
    return min(score, 1.0)

# --- CLI / Runner ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True, help="Folder of inputs (pdf/txt/html)")
    ap.add_argument("--glob", type=str, default="*.pdf", help="Glob pattern inside inputs")
    ap.add_argument("--labels", type=str, default=None, help="JSONL of gold labels (optional)")
    ap.add_argument("--optimize", action="store_true", help="Run teleprompter to tune the pipeline on labels")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    args = ap.parse_args()

    paths = sorted(Path(args.inputs).glob(args.glob))
    if not paths:
        rprint(f"[red]No files matched {args.inputs}/{args.glob}[/red]")
        return

    # Optional: load gold labels for optimization
    train_examples: List[Example] = []
    if args.labels and Path(args.labels).exists():
        with open(args.labels, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Expected fields: {"source": str, "paper_text"?: str, "gold": dict}
                paper_text = item.get("paper_text")
                if not paper_text:
                    src = item.get("source")
                    if src and Path(src).exists():
                        paper_text = load_text(Path(src))
                if not paper_text:
                    continue
                gold = item.get("gold", {})
                train_examples.append(Example(paper_text=paper_text, schema_hint=json.dumps({}), result_json=json.dumps(gold)))

    pipeline = PaperIE()

    if args.optimize and train_examples:
        rprint("[bold green]Optimizing with BootstrapFewShot + RandomSearch on labels...[/bold green]")
        tele = build_teleprompter(train_examples)
        tele.compile(pipeline, trainset=train_examples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for p in paths:
            try:
                text = load_text(p)
                # Optional: mini-chunk to mitigate context spikes
                text = text[:1_000_000]  # keep under 1M tokens equivalent; adjust as needed
                pred = pipeline(text)
                pred["paper_title"] = pred.get("paper_title") or p.stem
                pred["source"] = str(p)
                wf.write(json.dumps(pred, ensure_ascii=False) + "\n")
                rprint(f"[cyan]✓ Extracted[/cyan] {p.name}")
            except Exception as e:
                rprint(f"[red]✗ Failed[/red] {p.name}: {e}")

    rprint(f"[bold]Done. Wrote[/bold] {out_path}")

if __name__ == "__main__":
    main()
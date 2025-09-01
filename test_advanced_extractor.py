#!/usr/bin/env python3
"""
Test script for the advanced DSPy paper extractor
"""
import os
import json
import dspy
from pathlib import Path

# Configure DSPy with Gemini
lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
dspy.configure(lm=lm)

# Simplified DSPy signatures for testing
class ExtractResearchFacts(dspy.Signature):
    """Extract key research facts from a paper text."""
    paper_text = dspy.InputField(desc="The full paper text")
    ablations = dspy.OutputField(desc="Ablation studies and their results")
    key_deltas = dspy.OutputField(desc="Key improvements over prior work")
    new_findings = dspy.OutputField(desc="Novel discoveries and contributions")
    training_config = dspy.OutputField(desc="Training setup details")
    model_params = dspy.OutputField(desc="training hyperparameters and architecture")
    costs = dspy.OutputField(desc="Training/inference costs if mentioned")

class PaperExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractResearchFacts)

    def forward(self, paper_text: str):
        return self.extract(paper_text=paper_text)

def main():
    # Load sample paper
    sample_path = Path("samples/sample_paper.txt")
    if not sample_path.exists():
        print("Sample paper not found!")
        return

    with open(sample_path, "r", encoding="utf-8") as f:
        paper_text = f.read()

    print("🔬 Testing Advanced DSPy Paper Extractor")
    print("=" * 50)

    # Initialize extractor
    extractor = PaperExtractor()

    # Extract information
    print("📖 Processing paper...")
    result = extractor(paper_text)

    print("\n📋 Extracted Information:")
    print("-" * 30)
    print(f"🔬 Ablations: {result.ablations}")
    print(f"📈 Key Deltas: {result.key_deltas}")
    print(f"💡 New Findings: {result.new_findings}")
    print(f"⚙️  Training Config: {result.training_config}")
    print(f"🧠 Model Params: {result.model_params}")
    print(f"💰 Costs: {result.costs}")

    # Save results
    output = {
        "paper_title": "Attention Is All You Need",
        "source": str(sample_path),
        "ablations": result.ablations,
        "key_deltas": result.key_deltas,
        "new_findings": result.new_findings,
        "training_config": result.training_config,
        "model_params": result.model_params,
        "costs": result.costs
    }

    with open("extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n✅ Results saved to extraction_results.json")

if __name__ == "__main__":
    main()
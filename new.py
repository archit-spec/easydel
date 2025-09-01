# requirements:
# pip install dspy-ai arxiv google-generativeai

import arxiv
import dspy
import dotenv

dotenv.load_dotenv()  # load environment variables from .env file   

# --- configure Gemini Flash 2.5 as backend ---
llm = dspy.LM(model="gemini/gemini-2.5-flash", max_output_tokens=2048)
dspy.configure(lm=llm)

# --- DSPy signature for structured extraction ---
class ExtractResearchInsights(dspy.Signature):
    """
    Extract critical research information from a given text.
    """
    text = dspy.InputField(desc="The research paper abstract or excerpt.")

    ablations = dspy.OutputField(desc="Details on ablation studies conducted.")
    key_deltas = dspy.OutputField(desc="Key improvements or differences compared to prior work.")
    new_findings = dspy.OutputField(desc="Novel discoveries or contributions.")
    training_configs = dspy.OutputField(desc="Training setups: hyperparameters, dataset, optimizer, scheduler.")
    params = dspy.OutputField(desc="Model parameters: size, layers, hidden dimensions, etc.")
    cost = dspy.OutputField(desc="Reported training cost (compute, time, GPUs, $ if available).")

# --- DSPy predictor ---
extractor = dspy.Predict(ExtractResearchInsights)

# --- fetch latest arXiv papers ---
def fetch_latest_arxiv(query="cat:cs.CL", max_results=3):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    return list(client.results(search))

# --- main pipeline ---
if __name__ == "__main__":
    # Test different categories: "cat:cs.CL" (NLP), "cat:cs.CV" (vision), "cat:cs.LG" (ML)
    categories = ["cat:cs.LG", "cat:cs.CL", "cat:cs.CV"]
    for category in categories:
        print(f"\n{'='*50} Testing {category} {'='*50}")
        papers = fetch_latest_arxiv(query=category, max_results=3)

        for paper in papers:
            print("\n========================================")
            print("📄 Title:", paper.title)
            print("🔗 PDF:", paper.pdf_url)
            print("🗓 Published:", paper.published)

            # extract structured insights from title, abstract, and metadata
            full_text = f"""Title: {paper.title}

Authors: {', '.join([author.name for author in paper.authors])}

Abstract: {paper.summary}

Categories: {', '.join(paper.categories) if paper.categories else 'N/A'}

Published: {paper.published}

DOI: {paper.doi if paper.doi else 'N/A'}"""
            res = extractor(text=full_text)

            print("\n--- Extracted Insights ---")
            print("Ablations:", res.ablations)
            print("Key Deltas:", res.key_deltas)
            print("New Findings:", res.new_findings)
            print("Training Configs:", res.training_configs)
            print("Params:", res.params)
            print("Cost:", res.cost)

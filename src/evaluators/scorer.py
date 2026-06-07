from typing import Dict


def compute_final_scores(results: list) -> Dict:
    """
    Takes list of evaluation results.
    Returns average scores per category per model.
    """

    scores = {
        "hf":   {"factual": [], "hallucination": [], "bias": [], "jailbreak": [], "multiturn_memory": []},
        "groq": {"factual": [], "hallucination": [], "bias": [], "jailbreak": [], "multiturn_memory": []},
    }

    for r in results:
        model    = r["model"]
        category = r["category"]
        score    = r["score"]

        if model in scores and category in scores[model]:
            scores[model][category].append(score)

    # Compute averages
    averages = {}
    for model, categories in scores.items():
        averages[model] = {}
        all_scores = []
        for category, vals in categories.items():
            avg = round(sum(vals) / len(vals), 2) if vals else 0.0
            averages[model][category] = avg
            all_scores.extend(vals)
        averages[model]["overall"] = round(
            sum(all_scores) / len(all_scores), 2
        ) if all_scores else 0.0

    return averages
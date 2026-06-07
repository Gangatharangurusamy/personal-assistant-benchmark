import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
logger = logging.getLogger(__name__)
os.makedirs("logs/charts", exist_ok=True)


def load_final_scores() -> dict:
    with open("logs/final_scores.json", "r") as f:
        return json.load(f)


def load_eval_results() -> list:
    results = []
    with open("logs/eval_results.jsonl", "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ─────────────────────────────────────────
# CHART 1: Radar Chart — overall comparison
# ─────────────────────────────────────────

def plot_radar_chart(scores: dict) -> str:
    categories = ["factual", "hallucination", "bias", "jailbreak", "multiturn_memory"]
    labels     = ["Factual", "Hallucination", "Bias", "Jailbreak", "Memory"]
    N          = len(categories)

    hf_vals   = [scores.get("hf",   {}).get(c, 0) for c in categories]
    groq_vals = [scores.get("groq", {}).get(c, 0) for c in categories]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    hf_vals   += hf_vals[:1]
    groq_vals += groq_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.plot(angles, hf_vals,   "o-", linewidth=2, color="#00d4ff", label="OSS (Qwen)")
    ax.fill(angles, hf_vals,   alpha=0.15, color="#00d4ff")

    ax.plot(angles, groq_vals, "o-", linewidth=2, color="#ff6b6b", label="Frontier (Groq)")
    ax.fill(angles, groq_vals, alpha=0.15, color="#ff6b6b")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", size=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2","4","6","8","10"], color="grey", size=8)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        labelcolor="white",
        facecolor="#1a1a2e",
        edgecolor="grey",
        fontsize=10
    )

    ax.set_title(
        "OSS vs Frontier — Performance Radar",
        color="white", size=13, pad=20
    )

    path = "logs/charts/radar_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  ✅ Radar chart saved → {path}")
    return path


# ─────────────────────────────────────────
# CHART 2: Bar Chart — category scores
# ─────────────────────────────────────────

def plot_bar_chart(scores: dict) -> str:
    categories = ["factual", "hallucination", "bias", "jailbreak", "multiturn_memory"]
    labels     = ["Factual", "Hallucin.", "Bias", "Jailbreak", "Memory"]

    hf_vals   = [scores.get("hf",   {}).get(c, 0) for c in categories]
    groq_vals = [scores.get("groq", {}).get(c, 0) for c in categories]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars1 = ax.bar(x - width/2, hf_vals,   width, label="OSS (Qwen)",     color="#00d4ff", alpha=0.85)
    bars2 = ax.bar(x + width/2, groq_vals, width, label="Frontier (Groq)", color="#ff6b6b", alpha=0.85)

    ax.set_ylim(0, 11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("Score (0-10)", color="white")
    ax.set_title("Category-wise Score Comparison", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("grey")
    ax.yaxis.label.set_color("white")

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.2,
            f"{bar.get_height():.1f}",
            ha="center", color="white", fontsize=9
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.2,
            f"{bar.get_height():.1f}",
            ha="center", color="white", fontsize=9
        )

    ax.legend(
        facecolor="#1a1a2e",
        edgecolor="grey",
        labelcolor="white",
        fontsize=10
    )

    path = "logs/charts/bar_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  ✅ Bar chart saved → {path}")
    return path


# ─────────────────────────────────────────
# CHART 3: Latency comparison
# ─────────────────────────────────────────

def plot_latency_chart(hf_latency: float, groq_latency: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    models   = ["OSS\n(Qwen 0.5B)", "Frontier\n(Groq Llama-70B)"]
    latency  = [hf_latency, groq_latency]
    colors   = ["#00d4ff", "#ff6b6b"]

    bars = ax.bar(models, latency, color=colors, width=0.4, alpha=0.85)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 30,
            f"{bar.get_height():.0f}ms",
            ha="center", color="white", fontsize=11
        )

    ax.set_ylabel("Avg Latency (ms)", color="white")
    ax.set_title("Average Response Latency", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("grey")
    ax.yaxis.label.set_color("white")

    path = "logs/charts/latency_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  ✅ Latency chart saved → {path}")
    return path


# ─────────────────────────────────────────
# CHART 4: Overall score summary
# ─────────────────────────────────────────

def plot_overall_scorecard(scores: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    models  = ["OSS (Qwen)", "Frontier (Groq)"]
    overall = [
        scores.get("hf",   {}).get("overall", 0),
        scores.get("groq", {}).get("overall", 0),
    ]
    colors = ["#00d4ff", "#ff6b6b"]

    bars = ax.bar(models, overall, color=colors, width=0.4, alpha=0.85)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}/10",
            ha="center", color="white", fontsize=12, fontweight="bold"
        )

    ax.set_ylim(0, 11)
    ax.set_ylabel("Overall Score", color="white")
    ax.set_title("Overall Performance Scorecard", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("grey")

    path = "logs/charts/scorecard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  ✅ Scorecard saved → {path}")
    return path


# ─────────────────────────────────────────
# GENERATE ALL CHARTS
# ─────────────────────────────────────────

def generate_all_charts(hf_latency: float = 2500, groq_latency: float = 600) -> dict:
    logger.info("\n📊 Generating charts...")
    scores = load_final_scores()

    return {
        "radar":     plot_radar_chart(scores),
        "bar":       plot_bar_chart(scores),
        "latency":   plot_latency_chart(hf_latency, groq_latency),
        "scorecard": plot_overall_scorecard(scores),
    }
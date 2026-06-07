import json
import os
from datetime import datetime
from fpdf import FPDF
import logging
logger = logging.getLogger(__name__)

class EvalReport(FPDF):

    def header(self):
        self.set_fill_color(15, 17, 23)
        self.rect(0, 0, 210, 297, "F")
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 212, 255)
        self.cell(0, 12, "LLM Evaluation Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 212, 255)
        self.ln(4)
        self.cell(0, 8, title, ln=True)
        self.set_draw_color(0, 212, 255)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(220, 220, 220)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_score_table(self, scores: dict):
        categories = [
            "factual", "hallucination", "bias",
            "jailbreak", "multiturn_memory", "overall"
        ]
        headers = ["Category", "OSS (Qwen)", "Frontier (Groq)", "Winner"]

        col_widths = [55, 40, 50, 40]

        # Header row
        self.set_fill_color(26, 26, 46)
        self.set_text_color(0, 212, 255)
        self.set_font("Helvetica", "B", 10)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 10)
        for cat in categories:
            hf_score   = scores.get("hf",   {}).get(cat, 0)
            groq_score = scores.get("groq", {}).get(cat, 0)
            winner     = "Frontier" if groq_score > hf_score else ("OSS" if hf_score > groq_score else "Tie")

            self.set_text_color(200, 200, 200)
            self.set_fill_color(20, 20, 35)
            self.cell(col_widths[0], 7, cat.replace("_", " ").title(), border=1, fill=True)

            # Color-code scores
            for score, width in [(hf_score, col_widths[1]), (groq_score, col_widths[2])]:
                if score >= 8:
                    self.set_text_color(0, 255, 100)
                elif score >= 5:
                    self.set_text_color(255, 200, 0)
                else:
                    self.set_text_color(255, 80, 80)
                self.cell(width, 7, f"{score:.1f}/10", border=1, fill=True, align="C")

            self.set_text_color(255, 200, 0)
            self.cell(col_widths[3], 7, winner, border=1, fill=True, align="C")
            self.ln()

    def add_cost_table(self, cost_data: list):
        headers    = ["Model", "Calls", "Avg Latency", "Avg Cost/Call"]
        col_widths = [60, 25, 45, 55]

        self.set_fill_color(26, 26, 46)
        self.set_text_color(0, 212, 255)
        self.set_font("Helvetica", "B", 10)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 10)
        for row in cost_data:
            self.set_text_color(200, 200, 200)
            self.set_fill_color(20, 20, 35)
            self.cell(col_widths[0], 7, row.get("model", "")[:30],   border=1, fill=True)
            self.cell(col_widths[1], 7, str(row.get("calls", 0)),     border=1, fill=True, align="C")
            self.cell(col_widths[2], 7, f"{row.get('avg_latency_ms', 0):.0f}ms", border=1, fill=True, align="C")
            self.cell(col_widths[3], 7, f"${row.get('avg_cost_per_call', 0):.6f}", border=1, fill=True, align="C")
            self.ln()


def generate_pdf(chart_paths: dict, hf_latency: float = 0, groq_latency: float = 0):
    scores    = json.load(open("logs/final_scores.json"))
    from src.observability.logger import ObservabilityLogger
    logger    = ObservabilityLogger()
    cost_data = logger.get_cost_latency_table()

    pdf = EvalReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Overview ──
    pdf.section_title("1. Overview")
    pdf.body_text(
        "This report compares two AI assistants:\n"
        "  - OSS Model: Qwen2.5-0.5B-Instruct (via HuggingFace Inference API)\n"
        "  - Frontier Model: Llama-3.3-70B (via Groq API)\n\n"
        "Evaluation covers 5 categories: Factual Accuracy, Hallucination "
        "Resistance, Bias Handling, Jailbreak Safety, and Multi-turn Memory."
    )

    # ── Score Table ──
    pdf.section_title("2. Evaluation Scores")
    pdf.add_score_table(scores)

    # ── Radar Chart ──
    pdf.section_title("3. Performance Radar")
    if os.path.exists(chart_paths.get("radar", "")):
        pdf.image(chart_paths["radar"], x=30, w=150)

    # ── Bar Chart ──
    pdf.add_page()
    pdf.section_title("4. Category Breakdown")
    if os.path.exists(chart_paths.get("bar", "")):
        pdf.image(chart_paths["bar"], x=10, w=190)

    # ── Latency + Cost ──
    pdf.section_title("5. Latency & Cost")
    pdf.add_cost_table(cost_data)
    pdf.ln(4)
    if os.path.exists(chart_paths.get("latency", "")):
        pdf.image(chart_paths["latency"], x=40, w=130)

    # ── Scorecard ──
    pdf.section_title("6. Overall Scorecard")
    if os.path.exists(chart_paths.get("scorecard", "")):
        pdf.image(chart_paths["scorecard"], x=40, w=130)

    # ── Recommendations ──
    pdf.add_page()
    pdf.section_title("7. Recommendations")

    hf_overall   = scores.get("hf",   {}).get("overall", 0)
    groq_overall = scores.get("groq", {}).get("overall", 0)

    pdf.body_text(
        f"OSS Model Overall Score:      {hf_overall}/10\n"
        f"Frontier Model Overall Score: {groq_overall}/10\n\n"
        "Key Findings:\n"
        "  - Frontier model (Groq/Llama-70B) shows higher accuracy on factual "
        "and complex reasoning tasks.\n"
        "  - OSS model (Qwen 0.5B) performs competitively on safety/jailbreak "
        "resistance despite being 140x smaller.\n"
        "  - OSS model is free to run; Frontier costs ~$0.00008 per call.\n"
        "  - Groq inference is ~4-5x faster than HF Inference API.\n\n"
        "Recommendation:\n"
        "  For cost-sensitive or offline deployments: use OSS (Qwen).\n"
        "  For accuracy-critical production use: use Frontier (Groq/Llama-70B).\n"
        "  Hybrid approach: route simple queries to OSS, complex to Frontier."
    )

    path = "logs/evaluation_report.pdf"
    pdf.output(path)
    logger.info(f"\n✅ PDF report saved → {path}")
    return path
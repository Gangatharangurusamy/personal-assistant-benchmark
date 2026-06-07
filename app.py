import gradio as gr
from dotenv import load_dotenv
from src.pipelines.conversation import ConversationAssistant
from src.observability.logger   import ObservabilityLogger
import logging
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log")
    ]
)

logger = logging.getLogger(__name__)

load_dotenv()

# ─────────────────────────────────────────
# GLOBAL ASSISTANTS + LOGGER
# ─────────────────────────────────────────

oss_assistant      = ConversationAssistant(adapter_type="hf")
frontier_assistant = ConversationAssistant(adapter_type="groq")
logger             = ObservabilityLogger()


# ─────────────────────────────────────────
# CHAT FUNCTIONS
# ─────────────────────────────────────────

def chat_oss(message: str, history: list):
    if not message.strip():
        return history, history, ""

    result     = oss_assistant.chat(message)
    response   = result["response"]
    meta       = result["model_response"]
    is_harmful = result["is_harmful"]

    display = f"[BLOCKED] {response}" if is_harmful else response

    meta_info = (
        f"\n\n---\n"
        f"Model: {meta['model_name']} | "
        f"Latency: {meta['latency_ms']}ms | "
        f"Tokens: {meta['tokens_input']}in/{meta['tokens_output']}out | "
        f"Cost: ${meta['cost_usd']}"
    )

    # NEW FORMAT — dicts instead of tuples
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": display + meta_info})

    return history, history, ""

def chat_frontier(message: str, history: list):
    if not message.strip():
        return history, history, ""

    result     = frontier_assistant.chat(message)
    response   = result["response"]
    meta       = result["model_response"]
    is_harmful = result["is_harmful"]

    display = f"[BLOCKED] {response}" if is_harmful else response

    meta_info = (
        f"\n\n---\n"
        f"Model: {meta['model_name']} | "
        f"Latency: {meta['latency_ms']}ms | "
        f"Tokens: {meta['tokens_input']}in/{meta['tokens_output']}out | "
        f"Cost: ${meta['cost_usd']}"
    )

    # NEW FORMAT
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": display + meta_info})

    return history, history, ""

def chat_both(message: str, oss_history: list, frontier_history: list):
    if not message.strip():
        return oss_history, oss_history, frontier_history, frontier_history, ""

    oss_history, oss_history, _               = chat_oss(message, oss_history)
    frontier_history, frontier_history, _     = chat_frontier(message, frontier_history)

    return oss_history, oss_history, frontier_history, frontier_history, ""

def reset_all():
    """Clear both assistants memory"""
    oss_assistant.reset()
    frontier_assistant.reset()
    return [], [], [], [], "Memory cleared for both assistants."

def get_stats():
    """Fetch live observability stats"""
    stats = logger.get_stats()
    table = logger.get_cost_latency_table()

    text = "=== LIVE STATS ===\n\n"

    for model, s in stats.items():
        label = "OSS (Qwen)" if model == "hf" else "Frontier (Groq)"
        text += f"{label}:\n"
        text += f"  Total Calls    : {s['total_calls']}\n"
        text += f"  Avg Latency    : {s['avg_latency_ms']}ms\n"
        text += f"  Total Cost     : ${s['total_cost_usd']}\n"
        text += f"  Total Tokens   : {s['total_tokens']}\n"
        text += f"  Harmful Blocked: {s['harmful_blocked']}\n\n"

    text += "=== COST + LATENCY TABLE ===\n\n"
    for row in table:
        text += (
            f"  {row['adapter']:<6} | "
            f"calls={row['calls']} | "
            f"avg_latency={row['avg_latency_ms']}ms | "
            f"avg_cost=${row['avg_cost_per_call']}\n"
        )

    return text


# ─────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────

with gr.Blocks(title="LLM Evaluation Harness") as demo:

    # ── Header ──
    gr.Markdown("""
    # LLM Evaluation Harness
    ### OSS (Qwen2.5-0.5B) vs Frontier (Llama-3.3-70B via Groq)
    Compare both assistants side by side. All calls are logged automatically.
    """)

    # ── Main Chat Tab ──
    with gr.Tabs():

        with gr.Tab("Side by Side"):
            gr.Markdown("Send the **same message** to both assistants at once.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### OSS Assistant\n`Qwen2.5-0.5B-Instruct`")
                    oss_chatbot = gr.Chatbot(
                        height=400,
                        label="OSS (Qwen)",
                    )
                    oss_state = gr.State([])

                with gr.Column():
                    gr.Markdown("### Frontier Assistant\n`Llama-3.3-70B via Groq`")
                    frontier_chatbot = gr.Chatbot(
                        height=400,
                        label="Frontier (Groq)",
                    )
                    frontier_state = gr.State([])

            with gr.Row():
                both_input = gr.Textbox(
                    placeholder="Type a message to send to BOTH assistants...",
                    label="Message",
                    scale=4,
                )
                both_btn = gr.Button("Send to Both", variant="primary", scale=1)

            both_btn.click(
                fn=chat_both,
                inputs=[both_input, oss_state, frontier_state],
                outputs=[oss_chatbot, oss_state, frontier_chatbot, frontier_state, both_input],
            )
            both_input.submit(
                fn=chat_both,
                inputs=[both_input, oss_state, frontier_state],
                outputs=[oss_chatbot, oss_state, frontier_chatbot, frontier_state, both_input],
            )

        # ── OSS Only Tab ──
        with gr.Tab("OSS Assistant"):
            gr.Markdown("### Chat with OSS only\n`Qwen2.5-0.5B-Instruct — Free, HuggingFace`")

            oss_only_chatbot = gr.Chatbot(height=450)
            oss_only_state   = gr.State([])

            with gr.Row():
                oss_input = gr.Textbox(
                    placeholder="Ask anything...",
                    label="Message",
                    scale=4,
                )
                oss_btn = gr.Button("Send", variant="primary", scale=1)

            oss_btn.click(
                fn=chat_oss,
                inputs=[oss_input, oss_only_state],
                outputs=[oss_only_chatbot, oss_only_state, oss_input],
            )
            oss_input.submit(
                fn=chat_oss,
                inputs=[oss_input, oss_only_state],
                outputs=[oss_only_chatbot, oss_only_state, oss_input],
            )

        # ── Frontier Only Tab ──
        with gr.Tab("Frontier Assistant"):
            gr.Markdown("### Chat with Frontier only\n`Llama-3.3-70B — Groq API`")

            frontier_only_chatbot = gr.Chatbot(height=450)
            frontier_only_state   = gr.State([])

            with gr.Row():
                frontier_input = gr.Textbox(
                    placeholder="Ask anything...",
                    label="Message",
                    scale=4,
                )
                frontier_btn = gr.Button("Send", variant="primary", scale=1)

            frontier_btn.click(
                fn=chat_frontier,
                inputs=[frontier_input, frontier_only_state],
                outputs=[frontier_only_chatbot, frontier_only_state, frontier_input],
            )
            frontier_input.submit(
                fn=chat_frontier,
                inputs=[frontier_input, frontier_only_state],
                outputs=[frontier_only_chatbot, frontier_only_state, frontier_input],
            )

        # ── Observability Tab ──
        with gr.Tab("Observability"):
            gr.Markdown("### Live Stats — All logged calls")

            stats_output = gr.Textbox(
                label="Stats",
                lines=20,
                interactive=False,
            )

            with gr.Row():
                refresh_btn = gr.Button("Refresh Stats", variant="primary")
                reset_btn   = gr.Button("Reset Memory",  variant="stop")

            reset_msg = gr.Textbox(label="Status", interactive=False)

            refresh_btn.click(fn=get_stats,  outputs=stats_output)
            reset_btn.click(
                fn=reset_all,
                outputs=[
                    oss_chatbot, oss_state,
                    frontier_chatbot, frontier_state,
                    reset_msg
                ]
            )

        # ── Eval Report Tab ──
        with gr.Tab("Evaluation Report"):
            gr.Markdown("### Run Full Evaluation + Generate PDF Report")

            eval_output = gr.Textbox(
                label="Evaluation Log",
                lines=25,
                interactive=False,
            )
            run_eval_btn = gr.Button(
                "Run Full Evaluation",
                variant="primary"
            )

            def run_full_eval():
                from src.pipelines.evaluation import EvaluationRunner
                from src.reporting            import generate_all_charts, generate_pdf

                log = ""
                runner = EvaluationRunner()

                import io, sys
                buffer = io.StringIO()
                old    = sys.stdout
                sys.stdout = buffer

                scores = runner.run()
                charts = generate_all_charts(
                    hf_latency   = logger.get_stats().get("hf",   {}).get("avg_latency_ms", 2500),
                    groq_latency = logger.get_stats().get("groq", {}).get("avg_latency_ms", 600),
                )
                generate_pdf(charts)

                sys.stdout = old
                log = buffer.getvalue()
                log += "\n\nPDF saved to logs/evaluation_report.pdf"
                return log

            run_eval_btn.click(fn=run_full_eval, outputs=eval_output)

    # ── Footer ──
    gr.Markdown("""
    ---
    Built with LangGraph + HuggingFace + Groq
    """)


# ─────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
        )
    )
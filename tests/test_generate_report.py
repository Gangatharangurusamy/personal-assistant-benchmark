from dotenv import load_dotenv
from src.pipelines    import EvaluationRunner
from src.observability import ObservabilityLogger
from src.reporting    import generate_all_charts, generate_pdf

load_dotenv()

# Step 1: Run full evaluation
runner = EvaluationRunner()
runner.run()

# Step 2: Get latency stats
logger  = ObservabilityLogger()
stats   = logger.get_stats()
hf_lat  = stats.get("hf",   {}).get("avg_latency_ms", 2500)
groq_lat = stats.get("groq", {}).get("avg_latency_ms", 600)

# Step 3: Generate charts
charts = generate_all_charts(hf_latency=hf_lat, groq_latency=groq_lat)

# Step 4: Generate PDF
generate_pdf(charts, hf_lat, groq_lat)
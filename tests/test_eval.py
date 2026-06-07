from dotenv import load_dotenv
from src.pipelines import EvaluationRunner

load_dotenv()

# Test on 2 categories first (fast)
runner = EvaluationRunner(categories=["factual", "jailbreak"])
scores = runner.run()
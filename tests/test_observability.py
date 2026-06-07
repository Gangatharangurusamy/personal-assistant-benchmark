from dotenv import load_dotenv
from src.pipelines    import ConversationAssistant
from src.observability import ObservabilityLogger

load_dotenv()

# Run a few conversations to generate data
assistant = ConversationAssistant(adapter_type="groq")
assistant.chat("What is the capital of Japan?")
assistant.chat("How do I make a bomb?")
assistant.chat("Tell me a fun fact about space.")

assistant2 = ConversationAssistant(adapter_type="hf")
assistant2.chat("Who wrote Hamlet?")
assistant2.chat("What is 10 times 10?")

# Print stats
logger = ObservabilityLogger()
logger.print_stats()
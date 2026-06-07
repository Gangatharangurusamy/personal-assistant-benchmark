from dotenv import load_dotenv
from src.pipelines import ConversationAssistant

load_dotenv()

assistant = ConversationAssistant(adapter_type="groq")

print("\n--- Web Search ---")
r1 = assistant.chat("Search who is the current CM of Tamil Nadu")
print("FULL RESULT:", r1)          # ← print everything
print("RESPONSE:", r1["response"])

print("\n--- Calculator ---")
r2 = assistant.chat("Calculate 15% of 3500")
print("FULL RESULT:", r2)
print("RESPONSE:", r2["response"])

print("\n--- Memory Recall ---")
assistant.chat("My favourite color is blue")
r3 = assistant.chat("What did I say my favourite color was?")
print("FULL RESULT:", r3)
print("RESPONSE:", r3["response"])

print("\n--- Direct Answer ---")
r4 = assistant.chat("What is the capital of Japan?")
print("FULL RESULT:", r4)
print("RESPONSE:", r4["response"])
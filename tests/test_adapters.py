from dotenv import load_dotenv
from src.adapters import HFAdapter, GroqAdapter

load_dotenv()

system   = "You are a helpful assistant."
messages = [{"role": "user", "content": "What is 2+2?"}]

print("=== GROQ TEST ===")
groq = GroqAdapter(system_prompt=system)
print(groq.chat(messages))

print("=== HF TEST ===")
hf = HFAdapter(system_prompt=system)
print(hf.chat(messages))
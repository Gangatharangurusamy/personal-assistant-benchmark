from src.prompts import PromptRegistry

registry = PromptRegistry()
registry.summary()

# Get all factual prompts
factual = registry.get_category("factual")
print(factual[0])

# Get single prompt by ID
prompt = registry.get_prompt("jail_001")
print(prompt)
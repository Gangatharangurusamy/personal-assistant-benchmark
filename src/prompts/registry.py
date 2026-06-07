import json
import os
from typing import List, Dict, Optional
import logging
logger = logging.getLogger(__name__)

class PromptRegistry:

    def __init__(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "test_prompts.json"
        )
        with open(path, "r") as f:
            self.prompts = json.load(f)

    def get_category(self, category: str) -> List[Dict]:
        """Get all prompts from a category"""
        return self.prompts.get(category, [])

    def get_prompt(self, prompt_id: str) -> Optional[Dict]:
        """Get a single prompt by ID"""
        for category in self.prompts.values():
            for prompt in category:
                if prompt["id"] == prompt_id:
                    return prompt
        return None

    def get_all(self) -> Dict:
        """Get every prompt"""
        return self.prompts

    def get_categories(self) -> List[str]:
        """List all category names"""
        return list(self.prompts.keys())

    def summary(self):
        """logger.info a summary of the prompt bank"""
        logger.info("\n=== Prompt Registry ===")
        for category, prompts in self.prompts.items():
            logger.info(f"  {category:<20} → {len(prompts)} prompts")
        total = sum(len(v) for v in self.prompts.values())
        logger.info(f"  {'TOTAL':<20} → {total} prompts\n")
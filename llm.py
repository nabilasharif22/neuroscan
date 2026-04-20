
# ==========================================
# llm.py
# ==========================================

# 1. Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()

# 2. Get API key from .env
api_key = os.getenv("OPENAI_API_KEY")

# 3. Create OpenAI client  
from openai import OpenAI

client = OpenAI(api_key=api_key)

# ==========================================
# LLM EXTRACTION (MOCK VERSION)
# ==========================================

def extract_experiment_model(text):
    """
    Replace this with OpenAI API later
    """

    return {
        "experiments": [
            {
                "name": "Example Experiment",
                "manipulated_variables": ["dopamine"],
                "measured_variables": ["learning"],
                "model_links": [
                    {
                        "experiment_variable": "dopamine",
                        "model_component": "prediction error",
                        "relationship": "tests",
                        "confidence": 0.9
                    },
                    {
                        "experiment_variable": "neural activity",
                        "model_component": "belief",
                        "relationship": "correlates",
                        "confidence": 0.7
                    }
                ]
            }
        ]
    }
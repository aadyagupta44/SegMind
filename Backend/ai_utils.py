import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# print("Current working dir:", os.getcwd())   # <-- see where Python looks
# print("Files in this dir:", os.listdir(".")) # <-- check if .env is visible
import re

def clean_response(text: str) -> str:
    # Remove anything between <think>...</think>, including newlines
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Load environment variables
load_dotenv()
print("HF_API_KEY:", os.getenv("HF_API_KEY"))

# Initialize Hugging Face client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_API_KEY"],
)


# Define the marketing strategy prompt template
def build_prompt(customer_info):
    return f"""
You are a creative brand strategist. Based on the following customer information,
predict the customer segment and recommend the next marketing strategy. 

Do NOT include internal reasoning, analysis, or <think> sections.  
Only output the final structured response in the format below.

Customer Information:
{customer_info}

Your Response:
* Predicted Segment:
* Next Marketing Strategy (explained step by step):
"""

def generate_strategy(input_data):
    # Convert input dict into readable customer info string
    customer_info = "\n".join([f"{k}: {v}" for k, v in input_data.items()])

    # Build structured prompt
    prompt = build_prompt(customer_info)
    
    try:
        completion = client.chat.completions.create(
            model="HuggingFaceTB/SmolLM3-3B",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = completion.choices[0].message["content"]
        return clean_response(raw)
    except Exception as e:
        print("Error in generate_strategy:", e)
        return f"Error generating strategy: {e}"

   

# # Example usage
# if __name__ == "__main__":
#     input_data = {
#         "age": 19,
#         "gender": "Female",
#         "income": 5500,
#         "spending_score": 67,
#         "membership_years": 3,
#         "purchase_frequency": "Monthly",
#         "preferred_category": "Sports",
#         "last_purchase_amount": 120,
#     }

#     response = generate_strategy(input_data)
#     print("\n--- Marketing Strategy ---\n")
#     print(response)

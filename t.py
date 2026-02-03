validations:
  product_name:
    - id: "v1_brand_consistency"
      prompt: "Does the product name '{value}' follow the Brand guidelines? (No special characters, Title Case)."
    - id: "v2_length_check"
      prompt: "Is the product name '{value}' between 5 and 50 characters?"
  price:
    - id: "v3_currency_format"
      prompt: "Does the value '{value}' contain a recognizable currency symbol ($, €, £)?"
  description:
    - id: "v4_sentiment"
      prompt: "Is the product description '{value}' written in a positive, professional tone?"



import pandas as pd
import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import SelectionOutputParser
from typing import Dict

class QualityAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        # We use a simple instruction to force Yes/No
        self.output_instruction = "\n\nReturn ONLY the word 'Yes' or 'No'."

    def run_check(self, value, custom_prompt):
        full_prompt = custom_prompt + self.output_instruction
        formatted_prompt = full_prompt.format(value=value)
        
        response = self.llm.invoke(formatted_prompt)
        content = response.content.strip().lower()
        
        return "Yes" if "yes" in content else "No"

def process_quality_checks(df: pd.DataFrame, config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    agent = QualityAgent()
    results_df = df.copy()

    for column, checks in config['validations'].items():
        if column in df.columns:
            for check in checks:
                check_id = check['id']
                prompt = check['prompt']
                
                print(f"Running {check_id} on {column}...")
                results_df[check_id] = df[column].apply(lambda x: agent.run_check(x, prompt))
                
    return results_df



import seaborn as sns
import matplotlib.pyplot as plt

def visualize_results(results_df, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract all check IDs defined in YAML
    all_check_ids = []
    for checks in config['validations'].values():
        for c in checks:
            all_check_ids.append(c['id'])
            
    # Convert Yes/No to numeric for visualization
    summary_data = results_df[all_check_ids].replace({"Yes": 1, "No": 0})
    
    # Calculate pass percentage per check
    pass_rates = (summary_data.mean() * 100).to_frame().T
    pass_rates.index = ["Pass Rate (%)"]

    plt.figure(figsize=(12, 4))
    sns.heatmap(pass_rates, annot=True, cmap="RdYlGn", cbar=False, fmt=".1f")
    plt.title("Quality Validation Dashboard (Success % by Check ID)")
    plt.show()

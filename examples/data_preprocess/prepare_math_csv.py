
import os
import pandas as pd
from datasets import Dataset

def prepare_math_csv():
    input_path = "/home/ssmurali/mlmt/dummy_data/MATH.csv"
    output_dir = "/home/ssmurali/mlmt/data/mlmt/math"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Clean and reformat
    # We want: data_source, prompt, ability, reward_model, extra_info
    
    processed_data = []
    for idx, row in df.iterrows():
        question = str(row['question']).strip()
        ground_truth = str(row['answer']).strip()
        
        # Standard verl format for math
        # We don't add the instruction following suffix here because 
        # TrajectoryCollector will handle Turn 1/2/3 templates.
        item = {
            "data_source": "math_csv",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {
                "style": "rule", 
                "ground_truth": ground_truth
            },
            "extra_info": {
                "index": idx,
                "original_solution": str(row['correct_answer']).strip()
            }
        }
        processed_data.append(item)
    
    print(f"Processed {len(processed_data)} items.")
    
    # Convert to HuggingFace Dataset then to Parquet
    dataset = Dataset.from_list(processed_data)
    
    output_path = os.path.join(output_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    prepare_math_csv()






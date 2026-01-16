import os
import pandas as pd
from datasets import Dataset

def prepare_mbpp_csv():
    input_path = "/home/ssmurali/mlmt/dummy_data/mbpp.csv"
    output_dir = "/home/ssmurali/mlmt/data/mlmt/code"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {input_path}...")
    # Read the CSV. 
    df = pd.read_csv(input_path)
    
    processed_data = []
    for idx, row in df.iterrows():
        # MBPP columns: problem, reference_solution, test_list
        problem = str(row['problem']).strip()
        tests = str(row['test_list']).strip()
        
        # Standard verl format
        item = {
            "data_source": "mbpp",
            "prompt": [{"role": "user", "content": problem}],
            "ability": "code",
            "reward_model": {
                "style": "code", 
                "ground_truth": tests
            },
            "extra_info": {
                "index": idx,
                "problem": problem,
                "test_list": tests,
                "reference_solution": str(row['reference_solution']).strip()
            }
        }
        processed_data.append(item)
    
    print(f"Processed {len(processed_data)} items.")
    
    dataset = Dataset.from_list(processed_data)
    output_path = os.path.join(output_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    prepare_mbpp_csv()


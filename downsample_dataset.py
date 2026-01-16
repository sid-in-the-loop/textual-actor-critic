import pandas as pd
import os

input_path = "/home/ssmurali/mlmt/data/mlmt/math/train.parquet"
output_path = "/home/ssmurali/mlmt/data/mlmt/math/train_1020.parquet"

if not os.path.exists(input_path):
    print(f"❌ Error: {input_path} not found!")
else:
    df = pd.read_parquet(input_path)
    print(f"Original samples: {len(df)}")
    
    # Downsample to 1020 samples
    df_1020 = df.head(1020)
    df_1020.to_parquet(output_path)
    
    print(f"✅ Saved {len(df_1020)} samples to {output_path}")



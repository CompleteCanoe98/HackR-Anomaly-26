from datasets import load_dataset

# 1. Download the full dataset from Hugging Face
print("Downloading FEMNIST...")
dataset = load_dataset("flwrlabs/femnist")

# 2. Save the raw data to a local directory
local_path = "./local_femnist_data"
dataset.save_to_disk(local_path)
print(f"Dataset successfully saved to {local_path}")
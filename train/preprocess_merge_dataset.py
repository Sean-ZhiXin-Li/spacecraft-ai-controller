import numpy as np
import glob
import os

file_paths = glob.glob("E:/spacecraft_ai_project/data/data/dataset/expert_dataset_*.npy")
print("Matched files:", file_paths)

# Read and merge
all_data = []
for path in file_paths:
    arr = np.load(path)
    all_data.append(arr)
all_data = np.concatenate(all_data, axis=0)

# Save as a large file
os.makedirs("E:/spacecraft_ai_project/data/data/preprocessed", exist_ok=True)
np.save("E:/spacecraft_ai_project/data/data/preprocessed/merged_expert_dataset.npy", all_data)

print("Merged dataset saved.")

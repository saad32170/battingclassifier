import kagglehub

# Download latest version
path = kagglehub.dataset_download("aneesh10/cricket-shot-dataset")

print("Path to dataset files:", path)
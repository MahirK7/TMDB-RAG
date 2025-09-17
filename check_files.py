import kagglehub, os

path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")
print("Dataset folder:", path)

print("\nFiles inside:")
for f in os.listdir(path):
    print(f)

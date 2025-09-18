# Import the 'kagglehub' library, which allows you to download datasets from Kaggle
import kagglehub  

# Import the 'os' module, which provides functions to interact with the operating system (like working with files and directories)
import os  

# Use the kagglehub function 'dataset_download' to download the dataset
# The argument is the unique identifier (owner/dataset-name) of the dataset on Kaggle
# It returns the local path where the dataset has been downloaded
path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")

# Print the local folder path where the dataset has been saved
print("Dataset folder:", path)

# Print a blank line first, then the text "Files inside:" to show what’s inside the dataset folder
print("\nFiles inside:")

# Loop through every file/directory in the dataset folder (path)
for f in os.listdir(path):
    # Print the name of each file/directory inside the dataset folder
    print(f)

import os
import numpy as np

def check_unique_indices_in_npy_files(directory):
    unique_indices = set()  # Set to store unique indices
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            # Load the npy file
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            
            # Flatten the array (if it's multi-dimensional) and add to the set
            unique_indices.update(data.flatten())
    
    # Return the unique indices and their count
    return unique_indices, len(unique_indices)
import os
import numpy as np
from collections import Counter

def count_index_occurrences_in_npy_files(directory):
    index_counter = Counter()  # Counter to track index frequencies
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            # Load the npy file
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            
            # Flatten the array (if it's multi-dimensional) and update the counter
            index_counter.update(data.flatten())
    
    # Return the index frequencies
    return index_counter





# Example usage
directory_path = "/home/a3ilab01/treeai/mmsegmentation/predictions"  # Replace with your directory path
unique_indices, num_unique = check_unique_indices_in_npy_files(directory_path)

print(f"Unique indices found: {unique_indices}")
print(f"Total number of unique indices: {num_unique}")
index_counter = count_index_occurrences_in_npy_files(directory_path)
# Print the counts of each unique index
for index, count in index_counter.items():
    print(f"Index {index}: {count} occurrences")
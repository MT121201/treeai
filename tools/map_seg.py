import os
import numpy as np
from PIL import Image

def map_classes_in_annotation(annotation_dir, output_dir, mapping_dict):
    """
    Map classes from the original class indices (0-61) to the new class indices (0-56).
    
    :param annotation_dir: Directory containing the original annotation PNG files.
    :param output_dir: Directory where the remapped annotations will be saved.
    :param mapping_dict: A dictionary that maps original class indices (0-61) to new class indices (0-56).
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all annotation files in the provided directory
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".png"):
            # Construct full file path
            input_path = os.path.join(annotation_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Open the annotation PNG file
            annotation = Image.open(input_path)
            annotation_array = np.array(annotation)

            # Map each class index according to the mapping dictionary
            remapped_annotation = np.copy(annotation_array)
            for original_class, new_class in mapping_dict.items():
                remapped_annotation[annotation_array == original_class] = new_class

            # Save the remapped annotation as a new PNG file
            remapped_annotation_image = Image.fromarray(remapped_annotation.astype(np.uint8))
            remapped_annotation_image.save(output_path)

            print(f"Processed: {filename} -> saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Define the original-to-new class mapping dictionary (from 0-61 to 0-56)
    mapping_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
     6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
      14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
       21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27,
        29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34,
         36: 35, 37: 36, 38: 37, 39: 38, 40: 39, 41: 40, 42: 41,
          43: 42, 45: 43, 46: 44, 47: 45, 48: 46, 49: 47, 50: 48,
           52: 49, 53: 50, 55: 51, 56: 52, 58: 53, 59: 54, 60: 55}

    # Input and output directories
    annotation_dir = '/home/a3ilab01/treeai/dataset/segmentation/full/annotations_0/train'  # Update with the correct directory
    output_dir = '/home/a3ilab01/treeai/dataset/segmentation/full/remapped_annotation/val'  # Update with the desired output directory

    # Perform the class mapping
    map_classes_in_annotation(annotation_dir, output_dir, mapping_dict)

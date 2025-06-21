import os
import shutil
from tqdm import tqdm

def merge_datasets(source_dirs, output_dir):
    """
    Merges datasets from multiple source directories into a single output directory, organizing files by split and subfolder.
    For each source directory, this function copies files from the 'images', 'labels', and 'masks' subfolders within the 'train' and 'val' splits into the corresponding locations in the output directory. To avoid filename conflicts, each copied file is prefixed with the name of its source directory.
    Args:
        source_dirs (list of str): List of paths to the source dataset directories.
        output_dir (str): Path to the output directory where merged data will be stored.
    The resulting directory structure in output_dir will be:
        output_dir/
            images/
                train/
                val/
            labels/
                train/
                val/
            masks/
                train/
                val/
    Files that do not exist in a source directory for a given split/subfolder are skipped.
    """

    for split in ['train', 'val']:
        for subfolder in ['images', 'labels', 'masks']:
            os.makedirs(os.path.join(output_dir, subfolder, split), exist_ok=True)

    for source in source_dirs:
        for split in ['train', 'val']:
            for subfolder in ['images', 'labels', 'masks']:
                src_path = os.path.join(source, subfolder, split)
                dst_path = os.path.join(output_dir, subfolder, split)

                if not os.path.exists(src_path):
                    continue

                for fname in tqdm(os.listdir(src_path), desc=f"Merging {subfolder}/{split} from {source}"):
                    src_file = os.path.join(src_path, fname)

                    # Add prefix to avoid file name conflicts
                    prefix = os.path.basename(os.path.normpath(source))
                    new_fname = f"{prefix}_{fname}"
                    dst_file = os.path.join(dst_path, new_fname)

                    # Copy
                    shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    # Example usage
    dataset1 = 'dataset/12_RGB_SemSegm_640_fL'
    dataset2 = 'dataset/34_RGB_SemSegm_640_pL'
    output = 'dataset/merged_seg_dataset'

    merge_datasets([dataset1, dataset2], output)
    print("✅ Datasets merged successfully.")

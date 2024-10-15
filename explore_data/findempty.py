import os

def find_empty_subdirectories(path):
    empty_dirs = []
    for root, dirs, files in os.walk(path):
        if not dirs and not files:
            empty_dirs.append(root)
    return empty_dirs



# Example usage:
# target_path = '/lab/chengr_lab/12232381/code/DCASE2024/models_audio'
# empty_subdirectories = find_empty_subdirectories(target_path)
# print("Empty subdirectories:")
# for empty_dir in empty_subdirectories:
#     print(empty_dir)


import os
import shutil

def find_empty_directories(path):
    """Find all empty directories in the given path."""
    empty_dirs = []
    for root, dirs, files in os.walk(path):
        if not dirs and not files:
            empty_dirs.append(root)
    return empty_dirs

def delete_directories(directories):
    """Delete the specified directories."""
    for dir_path in directories:
        if os.path.exists(dir_path):
            print(f"Deleting: {dir_path}")
            shutil.rmtree(dir_path)

def find_and_delete_prefixed_dirs_in_path2(path2, prefix):
    """Delete directories in path2 that start with the same name as the prefix."""
    for root, dirs, files in os.walk(path2):
        for dir_name in dirs:
            if dir_name.startswith(prefix):
                full_path = os.path.join(root, dir_name)
                print(f"Deleting directory in path2: {full_path}")
                shutil.rmtree(full_path)

def delete_empty_dirs_in_path1_and_prefixed_in_path2(path1, path2):
    # Step 1: Find empty directories in path1
    empty_dirs_in_path1 = find_empty_directories(path1)

    # Step 2: Delete empty directories in path1
    delete_directories(empty_dirs_in_path1)

    # Step 3: Find and delete directories in path2 that start with the same prefix
    for empty_dir in empty_dirs_in_path1:
        dir_name = os.path.basename(empty_dir)  # Get the directory name
        find_and_delete_prefixed_dirs_in_path2(path2, dir_name)

# Example usage:
def find_directories_with_one_subdir_and_one_file(path):
    result_dirs = []
    
    for root, dirs, files in os.walk(path):
        # Check if the directory contains exactly one subdirectory and one file
        if len(dirs) == 1 and len(files) == 1:
            result_dirs.append(root)
    
    return result_dirs


if  __name__ == "__main__":
    # path1 = '/lab/chengr_lab/12232381/code/DCASE2024/models_audio'
    # path2 = '/lab/chengr_lab/12232381/code/DCASE2024/results_audio'
    # delete_empty_dirs_in_path1_and_prefixed_in_path2(path1, path2)
    target_path = '/lab/chengr_lab/12232381/code/DCASE2024/results_audio'
    directories = find_directories_with_one_subdir_and_one_file(target_path)
    delete_directories(directories)
    print("Directories with exactly one subdirectory and one file:")
    for directory in directories:
        print(directory)

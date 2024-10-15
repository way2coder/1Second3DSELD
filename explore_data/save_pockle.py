import pickle
import os

def save_list_to_file(file_list, file_path):
    """Save a Python list to a file using pickle."""
    with open(file_path, 'wb') as file:
        pickle.dump(file_list, file)

def remove_sub_suffix_and_store(directory):
    """Remove '_sub' suffix from filenames and store them in a list."""
    file_list = []
    number4 = 0
    number2 = 0
    # Get all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('_sub.npy'):
            # Remove '_sub' suffix and store the filename in the list
            new_filename = filename.replace('_sub', '')
            file_list.append(new_filename)
    
    # print( number2, number4)
    # Specify the path to save the list
    save_path = os.path.join(directory, 'random_validation_file')
    
    # Save the list to the file
    save_list_to_file(file_list, save_path)

# Example usage:
directory = '/lab/chengr_lab/12232381/dataset/STARSS2023/feat_label_hnet/foa_dev_gammatone_128bins_iv_7_norm'
remove_sub_suffix_and_store(directory)

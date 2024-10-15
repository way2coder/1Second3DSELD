import os
import random
import shutil
import re
from collections import defaultdict
import pickle
import shutil
''' 
statistic all the files in train and test norm of .npy and make random sample from these files in order to decrease the validation time but maintain the test time.
'''
# 文件夹路径

# 统计文件信息

# 遍历文件夹中的所有文件
def statistic_full_dataset(folder):
    statistics = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            # 使用正则表达式匹配文件名格式
            match = re.match(r'fold(\d+)_room(\d+)_mix(\d+)\.npy', filename)
            if match:
                fold_id = int(match.group(1))
                room_id = int(match.group(2))
                mix_id = int(match.group(3))
                
                # 将room_id和mix_id存入对应fold的列表中
                statistics[fold_id][room_id].append(filename)
    return statistics

def display_statistics(statistics):
    for fold_id, rooms in statistics.items():
        print(f"Fold {fold_id}:")
        for room_id, files in rooms.items():
            print(f"  Room {room_id} has {len(files)} files")

def select_files_and_save(folder_path, output_pickle_file):
    selected_files = []
    
    # Dictionary to store the files by fold and room
    files_by_fold_room = defaultdict(lambda: defaultdict(list))

    # Traverse the folder and categorize files by fold and room
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            match = re.match(r'fold(\d+)_room(\d+)_mix(\d+)\.npy', filename)
            if match:
                fold_id = int(match.group(1))
                room_id = int(match.group(2))
                files_by_fold_room[fold_id][room_id].append(filename)
    
    # Select 1/3 of the files from each room in fold1 and fold2
    for fold_id in [1, 2]:
        for room_id, files in files_by_fold_room[fold_id].items():
            num_files_to_select = len(files) // 3
            selected_files.extend(random.sample(files, num_files_to_select))
    # breakpoint()
    # Save the selected files to a pickle file
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(selected_files, f)

    print(f"Selected {len(selected_files)} files and saved them to {output_pickle_file}")


def copy_subset_to_newfolder(pickle_file, older_folder, new_folder): 
    with open(pickle_file, 'rb') as f:
        file_list = pickle.load(f)

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for filename in file_list:
        # Construct full file paths
        old_file_path = os.path.join(older_folder, filename)
        new_file_path = os.path.join(new_folder, filename)
        
        # Move the file to the new folder
        if os.path.exists(old_file_path):
            shutil.copy(old_file_path, new_file_path)
            print(f"Copied: {filename}")
        else:
            print(f"File not found: {filename}")
# Usage example

# # 随机抽取fold2中每个room的30个文件，并重命名
# if 2 in statistics:
#     for room_id, mixes in statistics[2].items():
#         if len(mixes) >= 30:
#             selected_files = random.sample(mixes, 30)
#             for filename in selected_files:
#                 # 新文件名
#                 new_filename = filename.replace('.npy', '_sub.npy')
#                 # 重命名文件
#                 shutil.move(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
#                 print(f"Renamed: {filename} -> {new_filename}")

# # 随机抽取fold4中的10个文件，并重命名
# if 4 in statistics and sum(len(mixes) for mixes in statistics[4].values()) >= 10:
#     all_files_fold4 = [file for room_files in statistics[4].values() for file in room_files]
#     selected_files_fold4 = random.sample(all_files_fold4, 10)
#     for filename in selected_files_fold4:
#         # 新文件名
#         new_filename = filename.replace('.npy', '_sub.npy')
#         # 重命名文件
#         shutil.move(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
#         print(f"Renamed: {filename} -> {new_filename}")
if __name__ == '__main__':
    folder_path = '/lab/chengr_lab/12232381/dataset/STARSS2023/feat_label_hnet/foa_dev_gammatone_128bins_iv_7_norm'
    new_folder = folder_path + '_subset'
    output_pickle_file = os.path.join(folder_path,'A_subset_files.pkl')
    copy_subset_to_newfolder(output_pickle_file, folder_path, new_folder)
    # select_files_and_save(folder_path, output_pickle_file)





 
import os
import shutil
import re

# 文件夹路径
import os

# 定义txt文件路
folder_path = '/lab/chengr_lab/12232381/dataset/STARSS2023/feat_label_hnet/foa_dev_gammatone_128bins_iv_7_norm/'  # 替换为包含npy文件的文件夹路径

# # 读取txt文件并处理内容
# with open(txt_file_path, 'r') as file:
#     lines = file.readlines()
# def function1()
#     for line in lines:
#         if "Renamed:" in line:
#             # 提取原始文件名和新文件名
#             parts = line.split("->")
#             original_name = parts[0].split(":")[1].strip()
#             new_name = parts[1].strip()
            
#             # 获取原始文件的完整路径和新的完整路径
#             original_file_path = os.path.join(folder_path, original_name)
#             new_file_path = os.path.join(folder_path, new_name)
            
#             # 如果文件存在，则重命名
#             if os.path.exists(new_file_path):
#                 os.rename(new_file_path, original_file_path)
#                 print(f"Renamed: {new_file_path} -> {original_file_path}")
#             else:
#                 print(f"File not found: {new_file_path}")


def change_name(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('_sub.npy'):
            # 去掉_sub后缀
            new_filename = filename.replace('_sub.npy', '.npy')
            
            # 将fold{i}替换为fold5
            # new_filename = re.sub(r'fold\d+', 'fold5', new_filename)
            
            # 重命名文件
            shutil.move(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f"Renamed: {filename} -> {new_filename}")

change_name(folder_path=folder_path)
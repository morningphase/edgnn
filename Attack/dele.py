import os
import glob

# 指定logs目录的路径
logs_dir = 'logs'

# 遍历logs目录下的所有子目录
for root, dirs, files in os.walk(logs_dir):
    # 搜索以'dgib_gnn_s_epoch'开头且扩展名为'.pt'的文件
    for file in glob.glob(os.path.join(root, 'dgib_gnn_s_epoch*.pt')):
        try:
            # 删除找到的文件
            os.remove(file)
            print(f"Deleted file: {file}")
        except OSError as e:
            print(f"Error: {e.strerror} : {file}")
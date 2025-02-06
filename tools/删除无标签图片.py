import os
import shutil

# 请在这里设置图片和标签文件夹的路径
IMAGE_DIR = r'打标文件/images'  # 替换为你的图片文件夹路径
LABEL_DIR = r'打标文件/labels'  # 替换为你的标签文件夹路径
TRASH_DIR = r'I:\Huge_project\pycharm\YOLO-8.3.32\软删除文件'  # 替换为你的回收站文件夹路径


def soft_delete_orphan_labels(image_dir, label_dir, trash_dir):
    """
    将没有对应图片的标签文件移动到回收站文件夹。

    参数:
        image_dir (str): 图片目录路径。
        label_dir (str): 标签目录路径。
        trash_dir (str): 回收站目录路径。
    """
    # 创建回收站文件夹（如果不存在）
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)

    # 获取图片和标签文件列表（不带扩展名）
    image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg')))
    label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt'))

    # 找出没有对应图片的标签文件
    orphan_labels = label_files - image_files

    # 将这些标签文件移动到回收站文件夹
    for label in orphan_labels:
        label_path = os.path.join(label_dir, label + '.txt')
        trash_path = os.path.join(trash_dir, label + '.txt')
        shutil.move(label_path, trash_path)
        print(f'Moved {label_path} to {trash_path}')

    print(f"清理完成。移动了 {len(orphan_labels)} 个无对应图片的标签文件到回收站。")


# 调用函数清理无对应图片的标签文件
if __name__ == "__main__":
    # 检查文件夹路径是否存在
    if not os.path.exists(IMAGE_DIR):
        print(f"错误：图片文件夹路径不存在 - {IMAGE_DIR}")
    elif not os.path.exists(LABEL_DIR):
        print(f"错误：标签文件夹路径不存在 - {LABEL_DIR}")
    else:
        soft_delete_orphan_labels(IMAGE_DIR, LABEL_DIR, TRASH_DIR)
import os
import shutil

# 请在这里设置图片和标签文件夹的路径
IMAGE_DIR = r'打标文件/images'  # 替换为你的图片文件夹路径
LABEL_DIR = r'打标文件/images'  # 替换为你的标签文件夹路径
TRASH_DIR = r'I:\Huge_project\pycharm\YOLO-8.3.32\软删除文件'  # 替换为你的回收站文件夹路径


def soft_delete_orphan_images(image_dir, label_dir, trash_dir):
    """
    将没有对应标签文件（.json）的图片文件移动到回收站文件夹。

    参数:
        image_dir (str): 图片目录路径。
        label_dir (str): 标签目录路径。
        trash_dir (str): 回收站目录路径。
    """
    # 创建回收站文件夹（如果不存在）
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)

    # 获取图片文件列表（不带扩展名）
    image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg')))

    # 获取标签文件列表（不带扩展名）
    label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.json'))

    # 找出没有对应标签文件的图片文件
    orphan_images = image_files - label_files

    # 将这些图片文件移动到回收站文件夹
    for image in orphan_images:
        # 查找图片文件的实际扩展名（可能是 .jpg, .png, .jpeg 等）
        for ext in ['.jpg', '.png', '.jpeg']:
            image_path = os.path.join(image_dir, image + ext)
            if os.path.exists(image_path):
                trash_path = os.path.join(trash_dir, image + ext)
                shutil.move(image_path, trash_path)
                print(f'Moved {image_path} to {trash_path}')
                break  # 找到后跳出循环

    print(f"清理完成。移动了 {len(orphan_images)} 个无对应标签文件的图片文件到回收站。")


# 调用函数清理无对应标签文件的图片文件
if __name__ == "__main__":
    # 检查文件夹路径是否存在
    if not os.path.exists(IMAGE_DIR):
        print(f"错误：图片文件夹路径不存在 - {IMAGE_DIR}")
    elif not os.path.exists(LABEL_DIR):
        print(f"错误：标签文件夹路径不存在 - {LABEL_DIR}")
    else:
        soft_delete_orphan_images(IMAGE_DIR, LABEL_DIR, TRASH_DIR)
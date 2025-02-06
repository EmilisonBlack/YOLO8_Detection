import os
import random
import shutil

# ----------------------
# 用户配置区域（手动修改以下路径）
# ----------------------
# 原始路径
TRAIN_IMAGES_DIR = "data/images/train"
TRAIN_LABELS_DIR = "data/labels/train"

# 目标路径
VAL_IMAGES_DIR = "data/images/val"
VAL_LABELS_DIR = "data/labels/val"

# 验证集比例 (0.2 表示20%)
VAL_RATIO = 0.15


# ----------------------

def validate_and_split():
    # 确保目标目录存在
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_LABELS_DIR, exist_ok=True)

    # 获取所有有效标签文件（排除没有对应图片的标签）
    valid_pairs = []
    for label_file in os.listdir(TRAIN_LABELS_DIR):
        if label_file.endswith(".txt"):
            # 获取不带扩展名的文件名
            base_name = os.path.splitext(label_file)[0]
            # 对应的图片文件路径
            img_path = os.path.join(TRAIN_IMAGES_DIR, f"{base_name}.jpg")

            # 检查图片是否存在
            if os.path.exists(img_path):
                valid_pairs.append((img_path, os.path.join(TRAIN_LABELS_DIR, label_file)))

    # 计算需要移动的数量
    total_valid = len(valid_pairs)
    val_count = int(total_valid * VAL_RATIO)
    print(f"找到 {total_valid} 个有效图片-标签对")
    print(f"需要移动 {val_count} 对到验证集")

    # 随机选择要移动的文件对
    selected_pairs = random.sample(valid_pairs, val_count)

    # 移动文件
    moved_count = 0
    for img_src, label_src in selected_pairs:
        try:
            # 移动图片
            img_dst = os.path.join(VAL_IMAGES_DIR, os.path.basename(img_src))
            shutil.move(img_src, img_dst)

            # 移动标签
            label_dst = os.path.join(VAL_LABELS_DIR, os.path.basename(label_src))
            shutil.move(label_src, label_dst)

            moved_count += 1
        except Exception as e:
            print(f"移动 {os.path.basename(img_src)} 失败: {str(e)}")

    print(f"\n操作完成！成功移动 {moved_count}/{val_count} 对文件到验证集")
    print(f"训练集剩余 {total_valid - moved_count} 对文件")


if __name__ == "__main__":
    validate_and_split()
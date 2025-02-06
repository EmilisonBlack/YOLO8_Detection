import os
import sys
import logging
import warnings
import cv2
import torch
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import shutil
from ultralytics import YOLO
from multiprocessing import get_context, Manager
from tqdm import tqdm
import numpy as np
import psutil
import time

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("image_processor.log"),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

##############################################################################
#                            配置参数（按需修改）                             #
##############################################################################

MODEL_PATH = "runs/best_model/exp24/weights/best.pt"  # 模型路径
DESTINATION_FOLDER = "I:/测试路径"  # 目标文件夹路径
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}  # 支持的图片格式
CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值 (0-1)
MAX_WORKERS = 12  # 最大并行进程数
BATCH_SIZE = 64  # 每批次处理图片数
MEMORY_LIMIT_GB = 24  # 内存限制（GB）
GPU_MEMORY_LIMIT = 0.8  # GPU显存使用上限


##############################################################################

class ImageProcessor:
    def __init__(self):
        self.target_classes = {}
        self.ctx = get_context('spawn')
        self.manager = Manager()
        self.total_images = 0
        self.moved_count = 0

    class MemoryController:
        """内存监控器"""

        def __init__(self):
            self.process = psutil.Process(os.getpid())

        @property
        def used_memory_gb(self):
            return self.process.memory_info().rss / (1024 ** 3)

        def is_memory_safe(self):
            system_mem = psutil.virtual_memory().percent < 90
            process_mem = self.used_memory_gb < MEMORY_LIMIT_GB * 0.9
            return system_mem and process_mem

    def _get_user_input(self):
        """获取用户输入（严格验证）"""
        root = tk.Tk()
        root.withdraw()

        # 获取模型类别（不加载完整模型）
        try:
            model_info = YOLO(MODEL_PATH, verbose=False).names
            class_names = list(model_info.values())
            del model_info  # 及时释放资源
        except Exception as e:
            messagebox.showerror("错误", f"模型信息获取失败: {str(e)}")
            sys.exit(1)

        target_classes = {}
        while True:
            class_name = simpledialog.askstring(
                "选择标签",
                f"可用类别: {class_names}\n输入类别名称（取消结束）:",
                parent=root
            )
            if not class_name:
                if not target_classes:
                    messagebox.showwarning("警告", "至少选择一个类别！")
                    continue
                break
            if class_name not in class_names:
                messagebox.showerror("错误", f"无效类别: {class_name}")
                continue
            try:
                min_count = simpledialog.askinteger(
                    "最小数量",
                    f"'{class_name}'的最小检测数量:",
                    minvalue=1,
                    initialvalue=1
                )
                if min_count is None:
                    raise ValueError
                target_classes[class_name] = min_count
            except:
                messagebox.showerror("错误", "请输入有效整数")

        src_folder = filedialog.askdirectory(title="选择图片文件夹")
        root.destroy()

        if not src_folder:
            logging.info("操作已取消")
            sys.exit(0)

        return target_classes, src_folder

    def _validate_image(self, img_path: str) -> bool:
        try:
            return cv2.imread(img_path) is not None
        except:
            return False

    @staticmethod
    def _worker_init(target_classes: dict):
        """独立进程初始化"""
        global worker_model
        try:
            # 初始化CUDA上下文
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_LIMIT)

            # 加载模型
            worker_model = YOLO(MODEL_PATH, verbose=False)
            if torch.cuda.is_available():
                worker_model.to('cuda')
            worker_model.fuse()

            # 验证类别兼容性
            model_classes = list(worker_model.names.values())
            missing = [k for k in target_classes if k not in model_classes]
            if missing:
                logging.error(f"模型缺少所需类别: {missing}")
                sys.exit(1)

        except Exception as e:
            logging.critical(f"进程初始化失败: {str(e)}")
            sys.exit(1)

    @staticmethod
    def _process_batch(args: tuple) -> list:
        """批量处理函数"""
        batch, target_classes = args
        move_files = []
        try:
            for img_path in batch:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    results = worker_model(img, imgsz=640, verbose=False)
                    counts = {k: 0 for k in target_classes}

                    if results and results[0].boxes is not None:
                        for det in results[0].boxes.data.cpu().numpy():
                            if len(det) < 6:
                                continue
                            *_, conf, cls_id = det
                            cls_name = worker_model.names.get(int(cls_id), "")
                            if cls_name in target_classes and conf >= CONFIDENCE_THRESHOLD:
                                counts[cls_name] += 1

                    if any(v >= target_classes[k] for k, v in counts.items()):
                        move_files.append(img_path)

                except Exception as e:
                    logging.error(f"处理失败 {os.path.basename(img_path)}: {str(e)}")
        except:
            pass
        return move_files

    def run(self):
        """主运行流程"""
        # 获取并验证输入
        self.target_classes, src_folder = self._get_user_input()
        logging.info(f"\n{'=' * 40}\n启动图片处理任务\n"
                     f"源文件夹: {src_folder}\n"
                     f"目标类别: {self.target_classes}\n"
                     f"内存限制: {MEMORY_LIMIT_GB}GB\n"
                     f"并行进程: {MAX_WORKERS}\n{'=' * 40}")

        # 收集有效图片
        all_images = [
            os.path.join(src_folder, f)
            for f in os.listdir(src_folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_FORMATS
        ]
        valid_images = [p for p in all_images if self._validate_image(p)]
        self.total_images = len(valid_images)

        if self.total_images == 0:
            logging.warning("未找到有效图片！")
            return

        # 准备批处理任务
        batches = [(valid_images[i:i + BATCH_SIZE], self.target_classes)
                   for i in range(0, self.total_images, BATCH_SIZE)]

        # 使用独立上下文管理进程池
        with self.ctx.Pool(
                processes=MAX_WORKERS,
                initializer=self._worker_init,
                initargs=(self.target_classes,)
        ) as pool:

            # 处理进度条
            results = []
            with tqdm(total=len(batches), desc="处理进度", unit="batch") as pbar:
                for batch in batches:
                    # 内存安全检查
                    while not self.MemoryController().is_memory_safe():
                        logging.warning("内存使用过高，暂停处理...")
                        time.sleep(5)

                    # 提交任务
                    res = pool.apply_async(
                        self._process_batch,
                        (batch,),
                        callback=lambda x: (results.extend(x), pbar.update())
                    )
                    try:
                        res.get(timeout=300)
                    except Exception as e:
                        logging.error(f"任务超时: {str(e)}")

            # 移动文件
            logging.info("\n开始移动符合要求的图片...")
            self.moved_count = 0
            with tqdm(total=len(results), desc="移动进度", unit="file") as pbar:
                for src_path in results:
                    try:
                        dest_path = os.path.join(DESTINATION_FOLDER, os.path.basename(src_path))
                        shutil.move(src_path, dest_path)
                        self.moved_count += 1
                        pbar.update()
                    except Exception as e:
                        logging.error(f"移动失败: {os.path.basename(src_path)} - {str(e)}")

            # 生成报告
            success_rate = self.moved_count / self.total_images * 100 if self.total_images > 0 else 0
            logging.info(
                f"\n{'=' * 40}\n"
                f"处理完成！\n"
                f"总图片数: {self.total_images}\n"
                f"成功移动: {self.moved_count}\n"
                f"成功率: {success_rate:.2f}%\n"
                f"{'=' * 40}"
            )


if __name__ == "__main__":
    try:
        processor = ImageProcessor()
        processor.run()
    except Exception as e:
        logging.critical(f"程序崩溃: {str(e)}", exc_info=True)
        sys.exit(1)
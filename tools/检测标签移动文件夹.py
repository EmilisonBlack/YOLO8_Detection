import os
import sys
import warnings
import cv2
import torch
import tkinter as tk
from tkinter import messagebox, filedialog
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from multiprocessing import get_context, Manager, Pool
from typing import Dict, List, Tuple
from collections import defaultdict
import psutil
import time


# 配置区域（针对Windows优化）
class Config:
    MODEL_PATH = "runs/best_model/exp24/weights/best.pt"
    DESTINATION = Path(r"I:\缓冲\漏批")
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    CONF_THRESH = 0.7
    MAX_WORKERS = 4  # 根据CPU核心数调整（建议物理核心数的50-75%）
    MAX_MEMORY_GB = 24  # 内存限制
    BATCH_SIZE = 8  # 初始批次大小
    MEMORY_SAFE_THRESHOLD = 0.9  # 内存使用安全阈值


warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


class WindowsMemoryController:
    """Windows专用内存控制器"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    @property
    def used_memory(self):
        """获取当前进程内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 ** 2

    def memory_safe(self):
        """检查内存是否安全"""
        total_used = psutil.virtual_memory().percent / 100
        process_used = self.used_memory / (Config.MAX_MEMORY_GB * 1024)
        return (total_used < Config.MEMORY_SAFE_THRESHOLD) and (process_used < 0.8)


class SharedModel:
    """共享模型加载器（Windows兼容方案）"""
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = YOLO(Config.MODEL_PATH).to("cuda")
            cls._model.fuse()
        return cls._model


class WindowsBatchProcessor:
    @staticmethod
    def worker_init():
        """工作进程初始化"""
        torch.cuda.empty_cache()
        SharedModel.get_model()  # 延迟加载

    @staticmethod
    def dynamic_batch_size(folder: Path, image_paths: List[Path]) -> int:
        """动态计算批次大小"""
        base_size = Config.BATCH_SIZE
        try:
            # 根据剩余内存调整批次
            vm = psutil.virtual_memory()
            available_mem = vm.available / (1024 ** 3)  # 转换为GB
            if available_mem < 2:
                return max(1, base_size // 4)
            elif available_mem < 4:
                return max(2, base_size // 2)
            return base_size
        except:
            return base_size

    @staticmethod
    def process_folder(args: Tuple[Path, List[Path], Dict]) -> Tuple[Path, Dict]:
        """处理单个文件夹"""
        folder_path, image_paths, target_classes = args
        counts = defaultdict(int)
        model = SharedModel.get_model()
        mem_ctrl = WindowsMemoryController()

        batch_size = WindowsBatchProcessor.dynamic_batch_size(folder_path, image_paths)
        total = len(image_paths)

        for i in range(0, total, batch_size):
            # 内存安全检查
            if not mem_ctrl.memory_safe():
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)

            batch_paths = image_paths[i:i + batch_size]
            images = []

            # 优化图像加载
            for p in batch_paths:
                try:
                    img = cv2.imread(str(p))
                    if img is not None:
                        img = cv2.resize(img, (640, 640))  # 统一尺寸减少内存占用
                        images.append(img)
                except Exception as e:
                    print(f"读取失败 {p.name}: {str(e)}")

            if not images:
                continue

            # 推理处理
            try:
                results = model(images, imgsz=640, verbose=False, stream=True)
                for result in results:
                    for detection in result.boxes.data.cpu().numpy():
                        *_, conf, cls_id = detection
                        if conf < Config.CONF_THRESH:
                            continue
                        class_name = result.names[int(cls_id)]
                        if class_name in target_classes:
                            counts[class_name] += 1
            except Exception as e:
                print(f"推理失败 {folder_path.name}: {str(e)}")

            # 及时释放资源
            del images
            torch.cuda.empty_cache()

        return (folder_path, counts)


class WindowsTaskManager:
    """Windows任务管理器"""

    def __init__(self):
        self.manager = Manager()
        self.lock = self.manager.Lock()
        self.task_queue = self.manager.Queue()
        self.results = self.manager.dict()
        self.mem_ctrl = WindowsMemoryController()

    def generate_tasks(self, folders: List[Path]):
        """生成内存安全的任务"""
        for folder in folders:
            image_paths = [
                p for p in folder.rglob("*")
                if p.suffix.lower() in Config.SUPPORTED_EXTS
                   and len(str(p)) < 200
            ]
            if image_paths:
                self.task_queue.put((folder, image_paths))

    def safe_get_task(self):
        """安全获取任务（带内存检查）"""
        while not self.task_queue.empty():
            if self.mem_ctrl.memory_safe():
                return self.task_queue.get()
            else:
                print("内存紧张，暂停任务分配...")
                torch.cuda.empty_cache()
                time.sleep(5)
        return None


class WindowsScanner:
    def __init__(self):
        self.target_classes = self._select_classes()
        self.task_manager = WindowsTaskManager()

    def _select_classes(self):
        """改进的类别选择逻辑"""
        model = YOLO(Config.MODEL_PATH)
        available_classes = list(model.names.values())
        print("可用的检测类别:", available_classes)

        target_classes = {}
        while True:
            cls = input("输入需要检测的类别名称（留空结束）: ").strip()
            if not cls:
                if not target_classes:
                    print("至少需要选择一个类别！")
                    continue
                break
            if cls not in available_classes:
                print(f"无效类别 '{cls}'，请从以下列表中选择: {available_classes}")
                continue
            try:
                min_count = int(input(f"设置 '{cls}' 的最小检测数量: "))
                target_classes[cls] = max(0, min_count)
            except ValueError:
                print("请输入有效的整数")
        return target_classes

    def scan(self, parent_folder: Path):
        """主扫描流程"""
        folders = [d for d in parent_folder.iterdir() if d.is_dir()]
        if not folders:
            messagebox.showinfo("完成", "没有找到子文件夹")
            return

        self.task_manager.generate_tasks(folders)

        # 创建进程池
        ctx = get_context("spawn")
        with ctx.Pool(
                processes=Config.MAX_WORKERS,
                initializer=WindowsBatchProcessor.worker_init
        ) as pool:

            # 进度条管理
            with tqdm(total=len(folders), desc="处理进度", unit="folder") as pbar:
                futures = []
                while True:
                    # 提交任务
                    while len(futures) < Config.MAX_WORKERS * 2:
                        task = self.task_manager.safe_get_task()
                        if task:
                            folder_path, image_paths = task
                            args = (folder_path, image_paths, self.target_classes)
                            fut = pool.apply_async(
                                WindowsBatchProcessor.process_folder,
                                args=(args,),
                                callback=lambda res: self._update_result(res, pbar)
                            )
                            futures.append(fut)
                        else:
                            break

                    # 检查完成情况
                    for fut in futures[:]:
                        if fut.ready():
                            futures.remove(fut)
                    if not futures and self.task_manager.task_queue.empty():
                        break
                    time.sleep(1)  # 这里需要已经导入time模块

        self._move_folders(folders)

    def _update_result(self, result: Tuple[Path, Dict], pbar):
        """更新结果"""
        folder_path, counts = result
        self.task_manager.results[str(folder_path)] = counts
        pbar.update(1)

    def _move_folders(self, folders: List[Path]):
        """安全移动文件夹"""
        moved = 0
        for folder in folders:
            counts = self.task_manager.results.get(str(folder), {})
            if all(counts.get(cls, 0) >= min_count for cls, min_count in self.target_classes.items()):
                dest = Config.DESTINATION / folder.name
                try:
                    shutil.move(str(folder), str(dest))
                    moved += 1
                except Exception as e:
                    print(f"移动失败 {folder.name}: {str(e)}")
        print(f"\n操作完成 | 成功移动 {moved}/{len(folders)} 个文件夹")


class WindowsApp:
    def run(self):
        """Windows专用主流程"""
        root = tk.Tk()
        root.withdraw()

        parent = Path(filedialog.askdirectory(title="选择父文件夹"))
        if not parent.is_dir():
            messagebox.showerror("错误", "无效的文件夹路径")
            return

        scanner = WindowsScanner()
        scanner.scan(parent)


if __name__ == "__main__":
    if sys.platform == "win32":
        from multiprocessing import freeze_support

        freeze_support()

    print(f"""
    {'*' * 40}
    Windows 10 优化方案:
    - 内存限制: {Config.MAX_MEMORY_GB}GB
    - 动态批次调整: 初始{Config.BATCH_SIZE}，根据内存自动调整
    - 安全阈值: 总内存使用 < {Config.MEMORY_SAFE_THRESHOLD * 100}%
    - 工作进程数: {Config.MAX_WORKERS}
    {'*' * 40}
    """)

    WindowsApp().run()
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('I:\\Huge_project\\pycharm\\YOLO-8.3.32\\runs\\best_model\\exp24\\weights\\best.pt')

# 导出为 ONNX 格式
model.export(format='onnx', imgsz=640, simplify=True)
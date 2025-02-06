# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：trian.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'I:\Huge_project\pycharm\YOLO-8.3.32\ultralytics\cfg\models\11\yolo11.yaml')
    #model.load('I:\Huge_project\pycharm\YOLO-8.3.32\yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'I:\Huge_project\pycharm\YOLO-8.3.32\data.yaml',
                imgsz=640,
                epochs=100,
                batch=32,
                workers=12,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )

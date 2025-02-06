# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'I:\Huge_project\pycharm\YOLO-8.3.32\runs\train\exp24\weights\best.pt')
    model.predict(source=r'I:\Huge_project\pycharm\YOLO-8.3.32\测试集',
                  save=True,
                  show=True,
                  )
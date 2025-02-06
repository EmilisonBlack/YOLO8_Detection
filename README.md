# YOLO11_Detection
基于YOLO11的检测色情内容模型（NSFW）
版权信息：本项目模板来源于CSDN博主 挂科边缘
原创内容：tools工具以及run文件夹下的模型

本模型专注于检测显露出来的乳房和下阴部分，软色情并不包含在内！

如何使用？
1.下载并解压我的项目
2.下载YOLO官方代码，与本项目合并，如有重复文件，则进行替换。
官方代码：https://github.com/ultralytics/ultralytics
（若不会操作，则可以参考该博主，最后将我的代码覆盖到下载的源码上|
博主地址：谷歌或百度搜索"YOLOv11来了，使用YOLOv11训练自己的数据集和推理(附YOLOv11网络结构图)"
）
3.搭建运行环境，安装依赖库。
4.解压 测试集\测试集.zip 中的图片
5.运行detect.py文件（注意要修改代码中的路径，改为你自己项目的路径）
6.查看模型精度是否满足自己需求。

TIP:tools工具里包含了一些应用工具，比如检测到乳房或者下阴到一定数量时，则会把该图片，或者该图片文件夹移动到指定路径，用于分类18+图片。（需要把项目地址改为自己电脑中的地址）

Copyright Information:
This project template originates from the CSDN blogger 挂科边缘.
Original content includes the tools utilities and the run folder models.

Model Description:
This model specializes in detecting exposed breasts and genital areas. Softcore content is not included!

How to Use?
Download and extract this project.
Download the official YOLO code and merge it with this project. Replace any duplicate files.
Official YOLO repository: https://github.com/ultralytics/ultralytics
If you're unsure how to proceed, refer to the tutorial by searching:
"YOLOv11来了，使用YOLOv11训练自己的数据集和推理(附YOLOv11网络结构图)"
on Google or Baidu. Then, overwrite the downloaded source code with this project's files.

Set up the runtime environment and install the required dependencies.
Extract images from TestSet/TestSet.zip.
Run the detect.py script, making sure to update the file paths in the code to match your local project directory.
Check the model's accuracy to see if it meets your requirements.
Additional Tools (TIP):
The tools folder contains some utility scripts.
For example, if the model detects a certain number of breast or genital regions in an image, it can automatically move the image (or its entire folder) to a designated directory for classifying 18+ content.

⚠ Make sure to update the project path in the script to match your local system.

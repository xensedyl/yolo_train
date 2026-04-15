# 安装
1. 模型下载地址:
https://github.com/ultralytics/assets/releases

ultralytics=8.3.93

# 训练
1. detect
```bash
pip install labeImg
```
labelImg
2. obb
```bash
cd model/roLabelImg
python roLabelImg.py
```
要退出advanced_mode
当前的ultralytics版本是8.3.161
3. sam
```bash
pip install labelme
```
labelme
自动保存，同时保存图像数据×，保留最后的标注
点击界面的创建多边形和编译多边形进行标注
# 使用
1. 在datasets中按照训练类别建立文件夹

2. 获取图片数据
```bash
python get_train_image.py
```
使用get_train_image()函数采集图片，需要设置开始的序号, 要采集的数量, 采样间隔时间

例如:采集图片100张，其中有30张图片重复或无效，剩余70张图片，但序号混乱，虽然使用rename_files()函数，将序号重置

有时会需要增加图片，可以使用rename_files_special()函数，将图片设置为从指定的序号开始


3. 修改对应的yaml文件，使用yolo_train.py进行训练

# 当前存在问题
1. yolo11n.pt等三个模型文件，放到model中，即使指定路径也无法使用

2. datasets中的yaml文件里，必须使用绝对路径，相对路径无法使用
# yolo_train # yolo_train

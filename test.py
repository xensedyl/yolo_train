import xml.etree.ElementTree as ET
import numpy as np
import math
import cv2
import io

# 模拟 XML 内容（你提供的那个）
xml_content = '''<annotation verified="yes">
    <folder>images</folder>
    <filename>00001</filename>
    <path>/home/li/backup/yolo_train/datasets/charge_obb/images/00001.png</path>
    <source><database>Unknown</database></source>
    <size><width>1280</width><height>720</height><depth>3</depth></size>
    <segmented>0</segmented>
    <object>
        <type>robndbox</type><name>charge</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
        <robndbox><cx>748.0585</cx><cy>308.1959</cy><w>101.3855</w><h>122.3984</h><angle>0.76</angle></robndbox>
    </object>
    <object>
        <type>robndbox</type><name>head</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
        <robndbox><cx>800.8925</cx><cy>252.1106</cy><w>35.6372</w><h>31.3782</h><angle>0.73</angle></robndbox>
    </object>
</annotation>'''

# 类别定义
classes = ["charge", "head"]

# 模拟读取 XML 文件
tree = ET.parse(io.StringIO(xml_content))
root = tree.getroot()

# 图像尺寸
width = int(root.find('size/width').text)
height = int(root.find('size/height').text)

# 输出归一化后的 OBB 标签
for obj in root.iter('object'):
    cls = obj.find('name').text
    if cls not in classes:
        continue
    cls_id = classes.index(cls)
    box = obj.find('robndbox')
    if box is None:
        continue

    cx = float(box.find('cx').text)
    cy = float(box.find('cy').text)
    w = float(box.find('w').text)
    h = float(box.find('h').text)
    angle = float(box.find('angle').text)

    Cos = np.cos(angle)
    Sin = np.sin(angle)
    vector1 = np.array([w / 2 * Cos, w / 2 * Sin])
    vector2 = np.array([-h / 2 * Sin, h / 2 * Cos])
    center = np.array([cx, cy])

    # 四个角点
    points = [
        center - vector1 - vector2,
        center + vector1 - vector2,
        center + vector1 + vector2,
        center - vector1 + vector2
    ]
    polys = np.array(points).flatten().tolist()

    # 标准矩形框角点
    xmin, ymin = min(polys[0::2]), min(polys[1::2])
    xmax, ymax = max(polys[0::2]), max(polys[1::2])
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    # 四种排序组合
    combine = [
        [[polys[0], polys[1]], [polys[2], polys[3]], [polys[4], polys[5]], [polys[6], polys[7]]],
        [[polys[2], polys[3]], [polys[4], polys[5]], [polys[6], polys[7]], [polys[0], polys[1]]],
        [[polys[4], polys[5]], [polys[6], polys[7]], [polys[0], polys[1]], [polys[2], polys[3]]],
        [[polys[6], polys[7]], [polys[0], polys[1]], [polys[2], polys[3]], [polys[4], polys[5]]]
    ]

    # 选择与标准矩形最接近的排序
    force, force_flag = float('inf'), 0
    for i, comb in enumerate(combine):
        temp_force = sum(math.sqrt(
            (comb[j][0] - dst_coordinate[j][0]) ** 2 + (comb[j][1] - dst_coordinate[j][1]) ** 2)
                         for j in range(4))
        if temp_force < force:
            force, force_flag = temp_force, i

    best_poly = np.array(combine[force_flag]).flatten().tolist()

    # 归一化
    normalized_poly = [
        best_poly[i] / width if i % 2 == 0 else best_poly[i] / height
        for i in range(len(best_poly))
    ]

    print(f"{cls_id} {' '.join(f'{x:.6f}' for x in normalized_poly)}")
# 0 0.588653 0.317935 0.646066 0.414944 0.580188 0.538165 0.522776 0.441156
# 1 0.623498 0.317412 0.644245 0.350420 0.627897 0.382895 0.607150 0.349888
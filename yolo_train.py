from ultralytics import YOLO
import xml.etree.ElementTree as ET
import numpy as np
import os,random,math,cv2,json,shutil
os.environ["WANDB_MODE"] = "dryrun"

class model_train:
    def __init__(self):
        # 定义目标文件夹路径
        datasets_name = "box4"
        self.data_path = f'datasets/{datasets_name}'
        self.labels_path = f"{self.data_path}/labels"
        self.images_path = f'{self.data_path}/images'
        self.xml_path = f"{self.data_path}/Annotations"
        self.yaml_path = f'{self.data_path}/{datasets_name}.yaml'
        # 定义类别,只有obb和seg需要
        self.classes = ["box4"]
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)

    def xml_to_boxtxt(self):
        for root, dirs, files in os.walk(self.xml_path):
            for file in files:
                if file.endswith(".xml"):
                    xml_path = os.path.join(root, file)
                    tree = ET.parse(xml_path)
                    root_elem = tree.getroot()

                    width = int(root_elem.find("size").find("width").text)
                    height = int(root_elem.find("size").find("height").text)

                    txt_file_name = os.path.splitext(file)[0] + ".txt"
                    txt_path = os.path.join(self.labels_path, txt_file_name)

                    with open(txt_path, "w") as txt_file:
                        for obj in root_elem.findall("object"):
                            name = obj.find("name").text
                            if name in self.classes:
                                class_id = self.classes.index(name)
                            else:
                                continue

                            bndbox = obj.find("bndbox")
                            xmin = int(bndbox.find("xmin").text)
                            ymin = int(bndbox.find("ymin").text)
                            xmax = int(bndbox.find("xmax").text)
                            ymax = int(bndbox.find("ymax").text)

                            x_center = (xmin + xmax) / (2.0 * width)
                            y_center = (ymin + ymax) / (2.0 * height)
                            width_bb = (xmax - xmin) / width
                            height_bb = (ymax - ymin) / height

                            txt_file.write(f"{class_id} {x_center} {y_center} {width_bb} {height_bb}\n")

    def xml_to_segtxt(self):
        # 参数设置
        xml_path, classes, labels_path = self.xml_path, self.classes, self.labels_path
        # 文件转换
        json_paths = os.listdir(xml_path)
        file_count = 0

        for json_path in json_paths:
            with open(os.path.join(xml_path, json_path), 'r') as load_f:
                json_dict = json.load(load_f)

            h, w = json_dict['imageHeight'], json_dict['imageWidth']
            txt_path = os.path.join(labels_path, json_path.replace('json', 'txt'))

            with open(txt_path, 'w') as txt_file:
                for shape_dict in json_dict['shapes']:
                    label_index = classes.index(shape_dict['label'])
                    points = shape_dict['points']
                    points_nor_str = ' '.join([f"{p[0] / w} {p[1] / h}" for p in points])
                    txt_file.write(f"{label_index} {points_nor_str}\n")
            file_count += 1
        print(f"处理完成的文件总数: {file_count}")

    def xml_to_obbtxt(self):
        xml_path, classes, labels_path, images_path = self.xml_path, self.classes, self.labels_path, self.images_path
        total_xml = [os.path.splitext(xml)[0] for xml in os.listdir(xml_path) if xml.endswith(".xml")]

        for image_id in total_xml:
            with open(os.path.join(xml_path, f"{image_id}.xml"), encoding='utf-8') as in_file:
                tree = ET.parse(in_file)
                root = tree.getroot()

                with open(f"{labels_path}/{image_id}.txt", 'w', encoding='utf-8') as list_file:
                    for obj in root.iter('object'):
                        cls = obj.find('name').text
                        if cls not in classes:
                            continue
                        cls_id = classes.index(cls)
                        box = obj.find('robndbox')
                        if box is None:
                            continue
                        cx, cy = float(box.find('cx').text), float(box.find('cy').text)
                        w, h, angle = float(box.find('w').text), float(box.find('h').text), float(
                            box.find('angle').text)
                        Cos, Sin = np.cos(angle), np.sin(angle)
                        vector1 = np.array([w / 2 * Cos, w / 2 * Sin])
                        vector2 = np.array([-h / 2 * Sin, h / 2 * Cos])
                        center = np.array([cx, cy])

                        points = [center - vector1 - vector2, center + vector1 - vector2, center + vector1 + vector2,
                                  center - vector1 + vector2]
                        polys = np.array(points).flatten().tolist()

                        xmin, ymin = min(polys[0::2]), min(polys[1::2])
                        xmax, ymax = max(polys[0::2]), max(polys[1::2])
                        dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

                        combine = [
                            [[polys[0], polys[1]], [polys[2], polys[3]], [polys[4], polys[5]], [polys[6], polys[7]]],
                            [[polys[2], polys[3]], [polys[4], polys[5]], [polys[6], polys[7]], [polys[0], polys[1]]],
                            [[polys[4], polys[5]], [polys[6], polys[7]], [polys[0], polys[1]], [polys[2], polys[3]]],
                            [[polys[6], polys[7]], [polys[0], polys[1]], [polys[2], polys[3]], [polys[4], polys[5]]]
                        ]

                        force, force_flag = float('inf'), 0
                        for i, comb in enumerate(combine):
                            temp_force = sum(math.sqrt(
                                (comb[j][0] - dst_coordinate[j][0]) ** 2 + (comb[j][1] - dst_coordinate[j][1]) ** 2) for
                                             j in range(4))
                            if temp_force < force:
                                force, force_flag = temp_force, i

                        best_poly = np.array(combine[force_flag]).flatten().tolist()
                        list_file.write(f"{cls_id} {' '.join(map(str, best_poly))}\n")
            # 读取并归一化TXT文件中的坐标信息
            txt_file_path = os.path.join(labels_path, f"{image_id}.txt")
            image_file_path = os.path.join(images_path, f"{image_id}.png")
            img = cv2.imread(image_file_path)

            if img is None:
                print(f"[警告] 图像读取失败: {image_file_path}")
                return

            height, width, _ = img.shape
            normalized_lines = []

            with open(txt_file_path, 'r') as f:
                for line in f:
                    parts = [float(x) for x in line.strip().split()]
                    if len(parts) != 9:
                        print(f"[跳过] 行格式不正确（不是1类+8坐标）: {line.strip()}")
                        continue
                    
                    category_index = int(parts[0])
                    coords = parts[1:]  # x1 y1 x2 y2 x3 y3 x4 y4
                    
                    # 归一化：x 除以 width, y 除以 height
                    normalized_coords = [
                        coords[i] / width if i % 2 == 0 else coords[i] / height
                        for i in range(len(coords))
                    ]
                    
                    normalized_line = [category_index] + normalized_coords
                    normalized_lines.append(normalized_line)

            # 写入归一化后的数据，覆盖原txt文件
            with open(txt_file_path, 'w') as f:
                for line in normalized_lines:
                    cls_id = int(line[0])
                    coords = line[1:]
                    formatted_coords = ' '.join(f"{x:.6f}" for x in coords)
                    f.write(f"{cls_id} {formatted_coords}\n")
        print(f"处理完成的文件总数: {len(total_xml)}")


    def process_dataset(self, train_ratio=0.9, val_ratio=0.1):
        labels_path, data_path, images_path = self.labels_path, self.data_path, self.images_path

        total_labels = os.listdir(labels_path)
        total_size = len(total_labels)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        trainval_indices = random.sample(range(total_size), train_size + val_size)
        train_indices = random.sample(trainval_indices, train_size)
        val_indices = set(trainval_indices) - set(train_indices)

        counts = {'train': 0, 'val': 0, 'test': 0}

        with open(f'{data_path}/train.txt', 'w') as train_file, \
                open(f'{data_path}/val.txt', 'w') as val_file, \
                open(f'{data_path}/test.txt', 'w') as test_file:

            for i, label_file in enumerate(total_labels):
                name = f'{images_path}/{label_file[:-4]}.png\n'  # Remove the .xml extension and format
                if i in train_indices:
                    train_file.write(name)
                    counts['train'] += 1
                elif i in val_indices:
                    val_file.write(name)
                    counts['val'] += 1
                else:
                    test_file.write(name)
                    counts['test'] += 1

        print(f"训练集数量: {counts['train']}, 验证集数量: {counts['val']}, 测试集数量: {counts['test']}")

    def train(self,mode):
        path = "model"
        if mode == "box":
            model = YOLO(model=f"yolo11n.pt")
        elif mode == "obb":
            model = YOLO(model=f"yolo11n-obb.pt")
        elif mode == "seg":
            model = YOLO(model=f"yolo11n-seg.pt")
        model.train(data=self.yaml_path, epochs=400)  # 训练模型

    def run(self):
        # print(self.yaml_path)
        # self.xml_to_boxtxt()
        # self.xml_to_obbtxt()
        # self.xml_to_segtxt()
        # self.process_dataset()
        self.train(mode = "box")

if __name__ == '__main__':
    m = model_train()
    m.run()




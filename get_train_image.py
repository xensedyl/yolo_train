import pyrealsense2 as rs
import numpy as np
import cv2,time,os


class yolo:
    def __init__(self):
        self.realsense_init()
        pass

    def realsense_init(self):
        # realsense API
        self.pipeline1 = rs.pipeline()  # 定义流程pipeline，创建一个管道 D415
        rs_config1 = rs.config()
        rs_config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        rs_config1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.align1 = rs.align(rs.stream.color)  # 创建对齐对象，以彩色图像为基准
        # rs_config1.enable_device('204222067870')  # d415
        cfg1 = self.pipeline1.start(rs_config1)
        self.intr1 = cfg1.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()  # 不同分辨率,内参不同
        self.depth_scale1 = cfg1.get_device().first_depth_sensor().get_depth_scale()  # 深度值和米的映射关系
        self.mtx1 = np.array([[self.intr1.fx, 0, self.intr1.ppx],
                              [0, self.intr1.fy, self.intr1.ppy],
                              [0, 0, 1]], dtype=np.float64)  # 相机内参

    def get_image1(self):
        frames = self.pipeline1.wait_for_frames()  # 原始图像集
        aligned_frames = self.align1.process(frames)  # 以彩色图为基准获取的对齐图像集
        color_frame = frames.get_color_frame()  # 从图像集中提取彩色图和深度图
        depth_frame1 = aligned_frames.get_depth_frame()
        self.color_image1 = np.asanyarray(color_frame.get_data())  # 图像np数组化
        self.depth_image1 = np.asanyarray(depth_frame1.get_data())

    def rename_files(self):
        folder_path = f"images"  # 替换为实际的文件夹路径
        png_files = sorted([file for file in os.listdir(folder_path) if file.endswith((".jpg",".png"))])
        for i, file in enumerate(png_files, start=1):
            new_filename = f"{i:05d}.png"
            old_filepath = os.path.join(folder_path, file)
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f'Renamed "{old_filepath}" to "{new_filepath}"')

    def rename_files_special(self):
        folder_path = 'images'  # 替换为实际的文件夹路径
        start_index,end_index,new_start_index = 1,58,73
        files = os.listdir(folder_path)
        # 使用列表推导式和条件表达式来筛选需要重命名的文件，并执行重命名操作
        renamed_files = [(file, os.path.splitext(file)[0]) for file in files
            if file.endswith('.png') and start_index <= int(os.path.splitext(file)[0]) <= end_index]
        for old_file, file_name in renamed_files:
            new_index = int(file_name) - start_index + new_start_index
            new_file_name = str(new_index).zfill(len(file_name))
            new_file_path = os.path.join(folder_path, f'{new_file_name}.png')
            os.rename(os.path.join(folder_path, old_file), new_file_path)
            print(f'Renamed "{old_file}" to "{new_file_name}.png"')

    def get_train_image(self):
        os.makedirs(f"images", exist_ok=True)
        save_floder,count,num,gap = f"images",0,10,0.3 # 开始的序号, 要采集的数量, 采样间隔时间
        count_end = count + num
        # 过滤无效图片
        for i in range(10):
            self.get_image1()
            time.sleep(0.1)
        while True:
            self.get_image1()
            img,filename = self.color_image1,f'{save_floder}/{count:05d}.png'
            ret = cv2.imwrite(filename, img)
            print(ret,filename)
            count += 1
            time.sleep(gap)
            cv2.namedWindow('realsense_cy', cv2.WINDOW_NORMAL)  # 创建一个窗口，让窗口能够缩放
            cv2.imshow('realsense_cy', img)  # 名字要相同
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            if count == count_end:
                break

    def run(self):
        self.get_train_image()
        # self.rename_files_special()
        # self.rename_files()
        pass




if __name__ == "__main__":
    y = yolo()
    y.run()

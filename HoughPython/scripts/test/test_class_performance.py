import cv2
import numpy as np
import time

img = cv2.imread(r"D:\CAVRide_LaneDetection\camera_calibration\Calibration_11.bmp")


class test01:
    def __init__(self):
        self.image = None

    def load_image(self, image):
        self.image = cv2.imread(image)

    def cleanup(self):
        self.image = None


class test02:
    def __init__(self):
        self.image = np.empty(shape=(1200, 1920, 3), dtype=np.uint8)

    def load_image(self, image):
        self.image = cv2.imread(image)

    def cleanup(self):
        self.image = np.empty(shape=(1200, 1920, 3), dtype=np.uint8)


img_path = r"D:\CAVRide_LaneDetection\camera_calibration\Calibration_11.bmp"

class_1 = test01()
class_2 = test02()
iterations = 1000
start_time = time.time()
for i in range(iterations):
    class_1.load_image(image=img_path)
    class_1.cleanup()
print(time.time() - start_time)

start_time = time.time()
for i in range(iterations):
    class_2.load_image(image=img_path)
    class_2.cleanup()
print(time.time() - start_time)

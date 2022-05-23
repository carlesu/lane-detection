import cv2
import numpy as np

import scripts.utils.image_utils as image_utils


class Frame:
    def __init__(self):
        self.original = None   # BGR
        self.processed = None  # BGR
        # Calibration Data
        self.cameraMatrix, self.distCoeffs = None, None
        # BEV data
        self.warp_matrix, self.inv_warp_matrix = None, None

    def load_image(self, image_array):
        self.original = image_array
        self.processed = image_array

    def load_calibration(self, cameraMatrix, distCoeffs):
        self.cameraMatrix, self.distCoeffs = cameraMatrix, distCoeffs

    def crop(self, top_margin=350, bottom_margin=127, left_margin=0, right_margin=0):
        row = self.processed.shape[0]
        col = self.processed.shape[1]
        self.processed = self.processed[top_margin:(row-bottom_margin), left_margin:(col-right_margin)]  # Crop image

    def gray(self):
        self.processed = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)

    def threshold(self):
        thresh, self.processed = cv2.threshold(self.processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def nieto(self, tau=50):
        self.processed = image_utils.nieto_filter_matrix(self.processed, tau)          # 0.016051530838012695 seconds
        # self.processed = image_utils.nieto_filter_loop(self.processed, tau)          # 2.445992946624756 seconds
        # self.processed = image_utils.nieto_filter_loop_veridic(self.processed, tau)  # 3.149050235748291 seconds

    def canny(self, low_threshold=200, high_threshold=300):
        self.processed = cv2.Canny(self.processed, low_threshold, high_threshold)

    def undistort(self):
        self.processed = cv2.undistort(self.processed, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)

    def BEV(self, src_points: np.ndarray, dest_points: np.ndarray):
        self.warp_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
        # shape = self.processed.shape
        shape = self.processed.shape[::-1]  # We invert the shape for some reason
        self.processed = cv2.warpPerspective(self.processed, self.warp_matrix, shape, flags=cv2.INTER_LINEAR)

    def show_processed(self, name='Frame'):
        cv2.imshow(name, self.processed)
        cv2.waitKey(0)

    def show_original(self):
        cv2.imshow('Frame', self.processed)
        cv2.waitKey(0)


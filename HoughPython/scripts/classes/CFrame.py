import cv2
import numpy as np

import scripts.utils.image_utils as image_utils


class Frame:
    cls_warp_matrix = None
    cls_inv_warp_matrix = None
    cls_cameraMatrix = None
    cls_distCoeffs = None
    cls_distMap1 = None
    cls_distMap2 = None

    def __init__(self, warp_matrix=None, inv_warp_matrix=None):
        self.original = None   # BGR
        self.processed = None  # BGR
        # Calibration Data
        self.cameraMatrix, self.distCoeffs = None, None
        # BEV data
        self.warp_matrix, self.inv_warp_matrix = None, None
        # Shape
        self.row, self.col = None, None

    def load_image(self, image_array):
        self.original = image_array.copy()
        self.row, self.col, _ = self.original.shape
        self.processed = image_array.copy()

    def load_calibration(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

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
        """
        The function is simply a combination of initUndistortRectifyMap (with unity R ) and remap
        (with bilinear interpolation).
        If we want performance, we shall only run initUndistortRectifyMap once and then use the remap.
        :return: 
        """
        if (self.cls_distMap1 is None) or (self.cls_distMap2 is None):


            print('Computing the undistortion and rectification transformation map.')  # TODO: Use logger
            R = None
            self.cls_distMap1, self.cls_distMap2 = cv2.initUndistortRectifyMap(self.cls_cameraMatrix,
                                                                               self.cls_distCoeffs,
                                                                               R,
                                                                               self.cls_cameraMatrix,
                                                                               self.processed.shape[0:2][::-1],
                                                                               cv2.CV_32FC1)
        print(self.processed.shape)
        self.processed = cv2.remap(self.processed, self.cls_distMap1, self.cls_distMap2, cv2.INTER_LINEAR)
        print(self.processed.shape)
        # self.processed = cv2.undistort(self.processed, self.cls_cameraMatrix, self.cls_distCoeffs, None, self.cls_cameraMatrix)
        # self.processed = cv2.undistort(self.processed, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)

    def BEV(self, src_points: np.ndarray, dest_points: np.ndarray):
        self.warp_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
        # shape = self.processed.shape
        shape = self.processed.shape[::-1]  # We invert the shape for some reason
        self.processed = cv2.warpPerspective(self.processed, self.warp_matrix, shape, flags=cv2.INTER_LINEAR)

    def show_processed(self, name='Processed', wait=0):
        cv2.imshow(name, self.processed)
        cv2.waitKey(int(wait))

    def show_original(self, name='Original', wait=0):
        cv2.imshow(name, self.processed)
        cv2.waitKey(int(wait))


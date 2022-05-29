import cv2
import numpy as np


class Frame:
    def __init__(self, logger):
        self.logger = logger
        self.original, self.processed = None, None  # BGR
        # Shape
        self.row, self.col, self.ch = None, None, None
        self.shape, self.inv_shape = None, None
        # Calibration
        self.camera_matrix, self.dist_coeffs = None, None
        # Distortion
        self.distMap1, self.distMap2 = None, None

    def load_dimensions(self, row, col, ch):
        self.row = row
        self.col = col
        self.ch = ch
        self.shape = (self.row, self.col)
        self.inv_shape = (self.col, self.row)

    def load_calibration(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def load_image(self, image_array):
        self.original = image_array
        self.processed = self.original.copy()

    def init_undistort_rectify_map(self):
        siz = self.inv_shape  # siz Undistorted image size [w,h].
        R = None
        self.distMap1, self.distMap2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, R,
                                                                   self.camera_matrix, siz, cv2.CV_32FC1)

    def gray(self):
        self.processed = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)

    def threshold(self):
        thresh, self.processed = cv2.threshold(self.processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def undistort_full(self):
        """
        The function is simply a combination of initUndistortRectifyMap (with unity R ) and remap
        (with bilinear interpolation).
        :return:
        """
        self.processed = cv2.undistort(self.processed, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

    def undistort_fast(self):
        """
        The function is simply a combination of initUndistortRectifyMap (with unity R ) and remap
        (with bilinear interpolation).
        If we want performance, we shall only run initUndistortRectifyMap once and then use the remap.
        * Computation increases from 0.06 to 0.011 seconds
        :return: 
        """
        self.processed = cv2.remap(self.processed, self.distMap1, self.distMap2, cv2.INTER_LINEAR)

    def show_processed(self, name='Processed', wait=0):
        cv2.imshow(name, self.processed)
        cv2.waitKey(int(wait))

    def show_original(self, name='Original', wait=0):
        cv2.imshow(name, self.processed)
        cv2.waitKey(int(wait))

    def cleanup_data(self):
        self.original = np.empty(shape=self.shape + (3, ), dtype=np.uint8)
        self.processed = np.empty(shape=self.shape + (3, ), dtype=np.uint8)

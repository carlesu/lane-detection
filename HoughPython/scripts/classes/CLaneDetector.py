import numpy as np
import cv2
import scripts.utils.image_utils as image_utils
from scripts.classes.CFrame import Frame


class LaneDetector(Frame):
    def __init__(self, logger):
        Frame.__init__(self, logger=logger)
        # ROI
        self.roi_top, self.roi_bottom, self.roi_left, self.roi_right = None, None, None, None
        self.roi_mask = None
        # BEV
        self.warp_matrix, self.inv_warp_matrix = None, None
        self.bev_src_points, self.bev_dest_points = None, None

    def init_perspective_transformation(self, bev_src_points, bev_dest_points):
        self.bev_src_points = bev_src_points
        self.bev_dest_points = bev_dest_points
        self.warp_matrix = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dest_points)
        self.inv_warp_matrix = cv2.getPerspectiveTransform(self.bev_dest_points, self.bev_src_points)

    def initialize_detector(self, camera_matrix, dist_coeffs, dimensions, roi_top, roi_bottom, roi_left, roi_right,
                            bev_src_points, bev_dest_points):
        # Set dimensions
        self.load_dimensions(row=dimensions[0], col=dimensions[1], ch=dimensions[2])
        # Load calibration
        self.load_calibration(camera_matrix=camera_matrix,
                              dist_coeffs=dist_coeffs)
        # Initialize undistort transformation
        self.init_undistort_rectify_map()
        # Initialize BEV transformation
        self.init_perspective_transformation(bev_src_points=bev_src_points, bev_dest_points=bev_dest_points)
        # Set ROI
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_left = roi_left
        self.roi_right = roi_right
        self.roi_mask = np.full(dimensions, False, dtype=bool)
        self.roi_mask[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right] = True

    def process(self, image_array):
        # Load the image
        self.load_image(image_array=image_array)
        # Undistort
        self.undistort_fast()
        # ROI
        # TODO
        # Nieto
        self.gray()
        self.nieto()
        # BEV
        self.BEV()

    def ROI(self):
        """
        Crop Region of Interest Mask.
        :return:
        """
        self.processed = self.processed[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]  # Crop image

    def canny(self, low_threshold=200, high_threshold=300):
        self.processed = cv2.Canny(self.processed, low_threshold, high_threshold)

    def nieto(self, tau=50):
        self.processed = image_utils.nieto_filter_matrix(self.processed, tau)          # 0.016051530838012695 seconds
        # self.processed = image_utils.nieto_filter_loop(self.processed, tau)          # 2.445992946624756 seconds
        # self.processed = image_utils.nieto_filter_loop_veridic(self.processed, tau)  # 3.149050235748291 seconds

    def BEV(self):
        # shape = self.processed.shape
        # shape = self.processed.shape[::-1]  # TODO We invert the shape for some reason
        self.processed = cv2.warpPerspective(self.processed, self.warp_matrix, self.inv_shape, flags=cv2.INTER_LINEAR)

    def cleanup_data(self):
        super(LaneDetector, self).cleanup_data()
        pass


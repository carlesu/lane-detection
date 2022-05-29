import cv2
import numpy as np
from scripts.classes.CChessboard import Chessboard


class CChessboardCalibration:
    def __init__(self, logger):
        self.logger = logger
        #
        self.image_size = None
        self.chessboard_dim = None
        #
        self.chessboard_images = list()
        self.chessboards = list()
        self.objects_points, self.images_points = list(), list()
        #
        self.retval, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = None, None, None, None, None
        self.calibration_data = dict()
        self.logger.info('Chessboard Calibrator created.')

    def load_images(self, chessboard_images: list, chessboard_dim: tuple):
        """
        All chessboard rows/cols shall be the same
        All image dimensions shall be the same
        :param chessboard_images: List of images (paths or image arrays(in BGR))
        :param chessboard_dim: (columns, rows) = (N of inner chessboard squares along the x-axis, N of inner chessboard
         squares along the y-axis)
        :return:
        """
        self.chessboard_images = chessboard_images
        self.chessboard_dim = chessboard_dim

    def build_chessboards(self):
        """
        Build every Chessboard object that will be used for further calibration.
        :return:
        """
        self.logger.info('Building {0} Chessboards'.format(len(self.chessboard_images)))
        # Initialize attributes
        self.chessboards = list()
        self.objects_points = list()
        self.images_points = list()
        # Build Chessboards
        for idx, chessboard_image in enumerate(self.chessboard_images):
            cur_chessboard = Chessboard(logger=self.logger, idx=idx,
                                        nx=self.chessboard_dim[0], ny=self.chessboard_dim[1])
            if cur_chessboard.load_image(image=chessboard_image):
                cur_chessboard.generate_points()
                if cur_chessboard.has_corners:
                    self.chessboards.append(cur_chessboard)
                    self.objects_points.append(cur_chessboard.object_points)
                    self.images_points.append(cur_chessboard.corners)
                else:
                    self.logger.warning('Chessboard has no corners. Skipping -> {0}'.format(chessboard_image))
            else:
                self.logger.warning('Chessboard not valid. Skipping -> {0}'.format(chessboard_image))
                continue
        self.logger.info('Total {0} Chessboards built.'.format(len(self.chessboards)))

    def calibrate(self):
        """
        Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
        objectPoints: It is a vector of vectors of calibration pattern points in the calibration pattern coordinate space
        imagePoints: It is a vector of vectors of the projections of calibration pattern point
        imageSize: Size of the image used only to initialize the camera intrinsic matrix.
        ---
        - retval: The overall RMS re-projection error.
        - cameraMatrix: Input/output 3x3 floating-point camera intrinsic matrix
        - distCoeffs: Input/output vector of distortion coefficients
        - rvecs: Output vector of rotation vectors (Rodrigues) estimated for each pattern view.
        - tvecs: Output vector of translation vectors estimated for each pattern view
        :return:
        """
        self.logger.info('Calibrating using {0} Chessboards.'.format(len(self.chessboards)))
        if len(self.chessboards):
            # Get the camera intrinsic matrix and distortion vector.
            self.image_size = self.chessboards[0].dimensions
            # Get the camera intrinsic matrix and distortion vector.
            self.retval, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.objects_points,
                self.images_points,
                self.image_size, None, None)
            self.calibration_data = dict(camera_matrix=self.camera_matrix, distortion_coefficient=self.dist_coeffs)
        else:
            self.retval, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = None, None, None, None, None
            self.calibration_data = None

    def get_calibration_data(self):
        return self.calibration_data

    def get_chessboards(self):
        return self.chessboards


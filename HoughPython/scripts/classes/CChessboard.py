import cv2
import time
import numpy as np


class Chessboard:
    def __init__(self, logger, nx, ny, idx=0):
        """
        TODO
        :param logger:
        :param nx: Number of inner chessboard squares along the x-axis
        :param ny: Number of inner chessboard squares along the y-axis
        """
        self.logger = logger
        self.idx = idx
        # Chessboard dimensions
        self.nx = nx
        self.ny = ny
        self.image, self.dimensions = None, None
        self.has_corners, self.corners = None, None
        self.object_points = None
        self.matrix, self.distortion = None, None

    def load_image(self, image):
        """
        Load the ChessBoard image
        :param image: List of images (paths or image arrays(in BGR))
        :return:
        """
        self.logger.debug('Loading Chessboard image...')
        if image is None:
            exception_str = 'No image was pased in the signature.'
            self.logger.warning(exception_str)
            return False
        else:
            if isinstance(image, str):
                try:
                    self.image = cv2.imread(image)  # In BGR
                    self.dimensions = self.image.shape[0:2]  # Shape returns (rows, cols, channels)
                    return True
                except:
                    exception_str = 'Could not read image: {0}'.format(image)
                    self.logger.critical(exception_str)
                    return False
            elif isinstance(image, np.ndarray):
                self.image = np.copy(image)
                self.dimensions = self.image.shape[0:2]  # Shape returns (rows, cols, channels)
                return True
            else:
                exception_str = 'Could not read image: {0}'.format(image)
                self.logger.warning(exception_str)
                return False

    def generate_points(self):
        """
        Generate 3D points (world coordinate frame)
        Generate 2D points (camera coordinate frame)
        :return:
        """
        self.logger.info('Processing Chessboard {0}...'.format(self.idx))
        start_time = time.time()
        self.find_corners()       # 2D Points
        self.get_object_points()  # 3D points
        self.logger.info('Processed in {0} seconds.'.format(time.time() - start_time))

    def find_corners(self):
        """
        https://learnopencv.com/camera-calibration-using-opencv/
        Finds the positions of internal corners of the chessboard.
        - image: Source chessboard view. It must be an 8-bit grayscale or color image.
        - patternSize: Number of inner corners per a chessboard row and column
        - flags: Various operation flags that can be zero or a combination. More information see cv2 doc.
        The function attempts to determine whether the input image is a view of the chessboard pattern and locate the
        internal chessboard corners. The function returns a non-zero value if all of the corners are found and they are
        placed in a certain order (row by row, left to right in every row). Otherwise, if the function fails to find all
        the corners or reorder them, it returns 0. For example, a regular chessboard has 8 x 8 squares and 7 x 7
        internal corners, that is, points where the black squares touch each other. The detected coordinates are
        approximate, and to determine their positions more accurately, the function calls cornerSubPix.
        You also may use the function cornerSubPix with different parameters if returned coordinates are not
        accurate enough.
        :return:
        """
        self.logger.debug('Calculating corners (2D)...')
        temp_image = np.copy(self.image)
        image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
        patternSize = (self.nx, self.ny)
        flags = None
        self.has_corners, self.corners = cv2.findChessboardCorners(image, patternSize, flags)
        # Corners are the location of the Chessboard Corners in 2D pixel coordinate frame.
        if self.has_corners:
            '''
            Find more exact corner pixels       
            Set termination criteria. We stop either when an accuracy is reached or when
            we have finished a certain number of iterations.
            '''
            self.logger.debug('Refining corner locations...')
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Termination criteria
            winSize = (11, 11)   # Size of the neighbourhood where it searches the corners.
            zeroZone = (-1, -1)  # Half of the neighbourhood size we want to reject. To not reject pass (-1, -1)
            self.corners = cv2.cornerSubPix(image, self.corners, winSize, zeroZone, criteria)

    def get_object_points(self):
        """
        Define real world coordinates for points in the 3D coordinate frame
        Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
        We need to define 3D world coordinates of the Chessboard corners, this shall be in concordance with the shape
        defined in the chessboard.
        Example: Chessboard of (3, 2) has 3 inner columns and 2 inner rows, a total of 6 corners. The real world 3D
        points shall be according to this [(0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,0), (2,1,0)]
        Note that the chessboard is in a plane, hence Z is always 0. The rest of coordinates are X,Y increasing with an
        equal step size; in this case we have chosen a step of 1.
        We could use a step of the shape of the squares for AX and AY.
        :return:
        """
        self.object_points = np.zeros((self.nx * self.ny, 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        # square_size = 0.023  # Length of the side of a square in meters
        # self.object_points = self.object_points * square_size

    def get_image(self):
        """
        Return the loaded image for the current ChessBoard object.
        :return:
        """
        return self.image

    def get_image_with_corners(self):
        """
        Draw 2D points (camera coordinate frame) in the loaded image.
        If this image doesn't have calculated corners, return raw image.
        :return: 
        """
        temp_image = np.copy(self.image)
        if self.has_corners:
            cv2.drawChessboardCorners(temp_image, (self.nx, self.ny), self.corners, self.has_corners)
        return temp_image

    def get_undistorted_image(self):
        """
        Undistort the loaded image using the calibration data.
        If camera calibration data is not initialized, return None
        :return:
        """
        if (self.matrix is not None) and (self.distortion is not None):
            temp_image = np.copy(self.image)
            return cv2.undistort(temp_image, self.matrix, self.distortion, None, self.matrix)
        else:
            return None

    def load_undistort_params(self, camera_matrix, distortion):
        """
        Load calibration data.
        :param camera_matrix: Camera intrinsic matrix
        :param distortion: Distortion coefficients
        :return:
        """
        self.distortion = distortion
        self.matrix = camera_matrix

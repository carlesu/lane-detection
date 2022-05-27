import cv2
import numpy as np


class ChessBoard:
    def __init__(self, nx=10, ny=7):
        """
        :param nx: Number of chessboard squares along the x-axis
        :param ny: Number of chessboard squares along the y-axis
        """
        # Chessboard dimensions
        self.nx = nx - 1  # Number of interior corners along x-axis
        self.ny = ny - 1  # Number of interior corners along y-axis
        self.image, self.dimensions = None, None
        self.has_corners, self.corners = None, None
        self.object_points = None
        self.matrix, self.distortion = None, None

    def load_image(self, image_array=None, image_path=None):
        """
        Load the ChessBoard image.
        :param image_array:
        :param image_path:
        :return:
        """
        if (image_array is None) and (image_path is None):
            raise Exception('No signature was passed.')
        else:
            if image_array is None:
                self.image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)  # In RGB
            elif image_path is None:
                self.image = np.copy(image_array)
            else:
                print('Both signatures image_array and image_path are passed. Using image_array.')
                self.image = np.copy(image_array)
            self.dimensions = self.image.shape[0:2]  # Shape returns (rows, cols, channels)

    def calibrate(self):
        """
        Generate 3D points (world coordinate frame)
        Generate 2D points (camera coordinate frame)
        :return:
        """
        print('ProcessingChessboard')  # TODO: Use loggger
        self.find_corners()       # 2D Points
        self.get_object_points()  # 3D points
        print('Processed')  # TODO: Use loggger

    def find_corners(self):
        """
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
        temp_image = np.copy(self.image)
        image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
        patternSize = (self.nx, self.ny)
        flags = None
        self.has_corners, self.corners = cv2.findChessboardCorners(image, patternSize, flags)
        if self.has_corners:
            '''
            Find more exact corner pixels       
            Set termination criteria. We stop either when an accuracy is reached or when
            we have finished a certain number of iterations.
            '''
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            winSize = (11, 11)   # Size of the neighbourhood where it searches the corners.
            zeroZone = (-1, -1)  # Half of the neighbourhood size we want to reject. To not reject pass (-1, -1)
            self.corners = cv2.cornerSubPix(image, self.corners, winSize, zeroZone, criteria)

    def get_object_points(self):
        """
        Define real world coordinates for points in the 3D coordinate frame
        Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
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
        Draw 2D points (camera coordinate frame) in the loaded iamge.
        If this image doesn't have calculated corners, return raw image.
        :return: 
        """
        temp_image = np.copy(self.image)
        if self.has_corners:
            cv2.drawChessboardCorners(temp_image, (self.nx, self.ny), self.corners, self.has_corners)
        return temp_image

    def get_undistorted_image(self):
        """
        Undistort the loaded image using the loaded undistort parameters.
        If camera parameters is not initialized, return None
        :return:
        """
        if (self.matrix is not None) and (self.distortion is not None):
            temp_image = np.copy(self.image)
            return cv2.undistort(temp_image, self.matrix, self.distortion, None, self.matrix)
        else:
            return None

    def load_undistort_params(self, camera_matrix, distortion):
        self.distortion = distortion
        self.matrix = camera_matrix

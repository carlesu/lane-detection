import scripts.utils.image_utils as image_utils
import scripts.utils.common_utils as common_utils
from scripts.classes.CChessboardCalibration import CChessboardCalibration
import cv2


logger = common_utils.create_logger()
calib_images = common_utils.get_multiple_files()

# Create the calibrator
chessboard_calibrator = CChessboardCalibration(logger=logger)
# Load the calibration images
# chessboard_dim = (N of inner chessboard squares along the x-axis, N of inner chessboard squares along the y-axis)
chessboard_calibrator.load_images(chessboard_images=calib_images, chessboard_dim=(7, 4))
# Build all chessboards infos
chessboard_calibrator.build_chessboards()
# Calibrate using all the Chessboards
chessboard_calibrator.calibrate()
# Get the data
calibration_data = chessboard_calibrator.get_calibration_data()
chessboards = chessboard_calibrator.get_chessboards()


# Testing plots
for chessboard in chessboards:
    if chessboard.has_corners:
        cv2.imshow("corners", chessboard.get_image_with_corners())
        cv2.waitKey(0)
        # Load the Calibration
        chessboard.load_undistort_params(camera_matrix=calibration_data['camera_matrix'],
                                         distortion=calibration_data['distortion_coefficient'])
        cv2.imshow("corners", chessboard.get_undistorted_image())
        cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imshow("corners", chessboard.get_image_with_corners())
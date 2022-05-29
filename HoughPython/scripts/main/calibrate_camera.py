import cv2
import scripts.utils.common_utils as common_utils
from scripts.classes.CChessboardCalibration import CChessboardCalibration

# Set constants
plot_result = False
save_data = True
chessboard_dim = (7, 4)  # (N of inner CB squares along the x-axis, N of inner CB squares along the y-axis)

# Create the logger
logger = common_utils.create_logger()
# Get the calibration images
calib_images = common_utils.get_multiple_files()

# Create the calibrator
chessboard_calibrator = CChessboardCalibration(logger=logger)
# Load the calibration images
chessboard_calibrator.load_images(chessboard_images=calib_images, chessboard_dim=chessboard_dim)
# Build all chessboards infos
chessboard_calibrator.build_chessboards()
# Calibrate using all the Chessboards
chessboard_calibrator.calibrate()
# Get the data
calibration_data = chessboard_calibrator.get_calibration_data()
chessboards = chessboard_calibrator.get_chessboards()

# Save data
if save_data:
    common_utils.save_pickle(data=calibration_data, save_path=r'./../../data/calibrations',
                             file_title='calibration_data')

# Testing plots
if plot_result:
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

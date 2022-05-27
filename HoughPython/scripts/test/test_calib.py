import scripts.utils.image_utils as image_utils
import scripts.utils.common_utils as common_utils
from scripts.classes.CChessBoard import ChessBoard
import cv2

calib_images = common_utils.get_multiple_files()

calibration_data, chessboards = image_utils.get_camera_calibration_matrix(chessboard_images=calib_images,
                                                                          chessboard_dim=(8, 5))

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
import scripts.utils.image_utils as image_utils
import scripts.utils.common_utils as common_utils
from scripts.classes.CChessBoard import ChessBoard
import cv2

calib_images = common_utils.get_multiple_files()

calibration_data, chessboards = image_utils.get_camera_calibration_matrix(chessboard_images=calib_images,
                                                                          chessboard_dim=(10, 7))

for chessboard in chessboards:
    if chessboard.has_corners:
        cv2.imshow("corners", chessboard.get_image_with_corners())
        cv2.waitKey(0)
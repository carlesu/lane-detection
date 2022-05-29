import cv2
import matplotlib.pyplot as plt
import scripts.utils.common_utils as common_utils


image_in = r"F:\CAVRide_LaneDetection\Exchange\camera_calibration\Calibration_09.bmp"
carlibration_file = r'../../data/calibrations/calibration_data.pkl'
calibration_data = common_utils.load_pickle(carlibration_file)

cameraMatrix = calibration_data['camera_matrix']
distCoeffs = calibration_data['distortion_coefficient']

chess_image = cv2.imread(image_in)  # In BGR
cv2.imshow("original", chess_image)
cv2.waitKey(0)

# Normal undistort
undist_0 = cv2.undistort(chess_image, cameraMatrix, distCoeffs, None, cameraMatrix)

# Optimized undistort
R = None
distMap1, distMap2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix,
                                                 chess_image.shape[0:2][::-1], cv2.CV_32FC1)
undist_1 = cv2.remap(chess_image, distMap1, distMap2, cv2.INTER_LINEAR)

cv2.imshow("undist_0", undist_0)

cv2.imshow("undist_1", undist_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

uneq_mask = ~(undist_0 == undist_1)

diff = (undist_0[uneq_mask].astype('int64') - undist_1[uneq_mask].astype('int64'))
max_diff = diff.max()
min_diff = diff.min()

plt.plot(diff, marker='.', linestyle='None')

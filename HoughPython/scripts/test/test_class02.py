import cv2
import numpy as np
import scripts.utils.common_utils as common_utils



video_file = common_utils.get_single_file()
carlibration_file = common_utils.get_single_file(file_type='pkl')
calibration_data = common_utils.load_pickle(carlibration_file)
# video_file = './../../data/vid/video20_001.avi'
capture = cv2.VideoCapture(video_file)
src_points_crop = np.array([(585, 61), (342, 243), (936, 243), (645, 61)], np.float32)
dest_points_crop = np.array([(342, 0), (342, 243), (936, 243), (936, 0)], np.float32)
src_points = np.array([(577, 416), (317, 599), (966, 604), (661, 423)], np.float32)
dest_points = np.array([(317, 0), (317, 720), (966, 720), (966, 0)], np.float32)


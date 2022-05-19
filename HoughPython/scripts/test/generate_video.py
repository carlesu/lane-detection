import scripts.utils.video_utils as video_utils
import scripts.utils.common_utils as common_utils

image_list = common_utils.get_multiple_files()
video_output = common_utils.file_save(file_extension='avi')
video_utils.images_to_video(image_list=image_list, video_output=video_output, fps=20)
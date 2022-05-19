import time
import cv2


def images_to_video(image_list, video_output, fps):
    """
    Assume all images have same lengthXwidth.
    :param image_list:
    :param video_output:
    :param fps:
    :return:
    """
    start_time = time.time()
    img1 = cv2.imread(image_list[0])
    height, width, layers = img1.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    for img_path in image_list:
        print('Processing frame {0}'.format(img_path))
        img = cv2.imread(img_path)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    print('Done in {0}'.format(time.time() - start_time))

    return


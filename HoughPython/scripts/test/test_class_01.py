import time
import cv2
from scripts.classes.Frame import Frame

capture = cv2.VideoCapture('./../../data/vid/video20_001.avi')

while 1:
    ret, image = capture.read()
    if image is None:
        break
    start_time = time.time()
    my_frame = Frame()
    my_frame.load_image(image_array=image)
    my_frame.crop()
    my_frame.gray()
    my_frame.nieto(tau=50)
    my_frame.canny()
    print("--- %s seconds ---" % (time.time() - start_time))
    # my_frame.show_processed()

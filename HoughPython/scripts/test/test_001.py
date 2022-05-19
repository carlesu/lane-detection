import time
import cv2

capture = cv2.VideoCapture('./../../data/vid/video20_001.avi')

while 1:
    start_time = time.time()
    ret, image = capture.read()
    if image is None:
        break
    row = image.shape[0]
    col = image.shape[1]
    image = image[350:(row-127), 0:col]  # Crop image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tau = 50
    nieto_image = nieto_filter_matrix(gray, tau)
    nieto_image = nieto_image.astype('uint8')
    thresh, th2 = cv2.threshold(nieto_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # kernel = np.ones((2, 2), np.uint8)
    # # image = cv2.erode(th2, kernel)
    # # image = cv2.dilate(image, kernel)
    # image = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

    # nieto_image = gray
    # print("--- %s seconds ---" % (time.time() - start_time)) # 0.006
    low_threshold = 200
    high_threshold = 300
    edges = cv2.Canny(nieto_image, low_threshold, high_threshold)
    print("--- %s seconds ---" % (time.time() - start_time))
    cv2.imshow('Frame', image)
    cv2.waitKey(1) # Time it is waiting between iterations
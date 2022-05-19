import cv2
import scripts.utils.image_utils as image_utils


class Frame:
    def __init__(self):
        self.original = None
        self.processed = None

    def load_image(self, image_array):
        self.original = image_array
        self.processed = image_array

    def crop(self, top_margin=350, bottom_margin=127, left_margin=0, right_margin=0):
        row = self.processed .shape[0]
        col = self.processed .shape[1]
        self.processed = self.processed[top_margin:(row-bottom_margin), left_margin:(col-right_margin)]  # Crop image

    def gray(self):
        self.processed = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)

    def nieto(self, tau=50):
        self.processed = image_utils.nieto_filter_matrix(self.processed, tau)

    def canny(self, low_threshold=200, high_threshold=300):
        self.processed = cv2.Canny(self.processed, low_threshold, high_threshold)

    def show_processed(self, name='Frame'):
        cv2.imshow(name, self.processed)
        cv2.waitKey(0)

    def show_original(self):
        cv2.imshow('Frame', self.processed)
        cv2.waitKey(0)


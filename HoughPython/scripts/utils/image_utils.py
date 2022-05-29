import numpy as np
import cv2


def normalize_gray(image_in: np.ndarray) -> np.ndarray:
    image = np.copy(image_in)
    min_value = image.min()
    max_value = image.max()
    row, col = image.shape
    normalized_gray_image = image / (max_value - min_value) - np.ones((row, col)) * min_value / (max_value - min_value)
    return normalized_gray_image


def adjust_gamma(image: np.ndarray, gamma=1.0):  # TODO: Return type¿
    # Input image shall be a numpy array grayscale image
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def color_selection(image: np.ndarray, red_threshold: float, green_threshold: float,
                    blue_threshold: float) -> np.ndarray:
    # Make a copy of the image
    color_select = np.copy(image)
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    # Do a boolean or with the "|" character to identify pixels below the thresholds
    thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (
                image[:, :, 2] < rgb_threshold[2])  # If any component below the threshold, discard it
    color_select[thresholds] = [0, 0, 0]
    return color_select


def nieto_filter_matrix(image_in: np.ndarray, tau: int) -> np.ndarray:
    """
    Optimized implementation with numpy arrays.
    Nieto, M., Arrospide, J., Salgado, L.: Road environment modeling using robust perspective analysis and recursive
    Bayesian segmentation.
    :param image_in:
    :param tau:
    :return:
    """
    image = np.copy(image_in)
    row, col = image_in.shape
    last_col = col - tau
    image = image.astype('int16')  # X_(i)
    left_matrix = np.zeros_like(image)   # Pre-allocate left operand
    right_matrix = np.zeros_like(image)  # Pre-allocate right operand
    left_matrix[:, [range(tau, col)]] = image[:, [range(last_col)]]   # X_(i-tau)
    right_matrix[:, [range(last_col)]] = image[:, [range(tau, col)]]  # X_(i+tau)
    # Y_(i) = 2*X_(i) - (X_(i-tau) + X_(i+tau)) - |X_(i-tau) - X_(i+tau)|
    nieto_image = (2 * image) - (left_matrix + right_matrix) - abs(left_matrix - right_matrix)
    nieto_image[nieto_image < 0] = 0
    nieto_image[nieto_image > 255] = 255
    nieto_image = nieto_image.astype('uint8')
    return nieto_image


def nieto_filter_loop(image_in: np.ndarray, tau: int) -> np.ndarray:
    """
    Original implementation from:
    Nieto, M., Arrospide, J., Salgado, L.: Road environment modeling using robust perspective analysis and recursive
    Bayesian segmentation.
    :param image_in:
    :param tau:
    :return:
    """
    image = np.copy(image_in)
    row, col = image.shape
    image = image.astype('int16')
    last_col = col - tau
    nieto_image = np.empty((0, col), 'int16')
    for i in range(row):
        row = np.empty(col, 'int16')
        for j in range(col):
            if j < tau:
                left_side = 0
                right_side = image[i][j + tau]
            elif (j + tau + 1) > col:
                right_side = 0
                left_side = image[i][j - tau]
            else:
                right_side = image[i][j + tau]
                left_side = image[i][j - tau]
            aux = 2 * image[i][j] - (left_side + right_side) - abs(left_side - right_side)
            if aux < 0:
                aux = 0
            elif aux > 255:
                aux = 255
            row[j] = aux
        nieto_image = np.vstack([nieto_image, row])
    nieto_image = nieto_image.astype('uint8')
    return nieto_image


def nieto_filter_loop_2(image_in: np.ndarray, tau: int) -> np.ndarray:
    """
    Original implementation from:
    Nieto, M., Arrospide, J., Salgado, L.: Road environment modeling using robust perspective analysis and recursive
    Bayesian segmentation.
    :param image_in:
    :param tau:
    :return:
    """
    image = np.copy(image_in)
    row, col = image.shape
    image = image.astype('int16')

    nieto_image = np.zeros((row, col), 'int16')
    for j in range(row):
        in_row = image[j]
        for i in range(tau, col - tau):
            if in_row[i] != 0:
                aux = 2 * in_row[i]
                aux += -in_row[i-tau]
                aux += -in_row[i+tau]
                aux += -abs(in_row[i-tau] - in_row[i+tau])
                if aux < 0:
                    aux = 0
                elif aux > 255:
                    aux = 255
                nieto_image[j][i] = aux
    nieto_image = nieto_image.astype('uint8')

    return nieto_image


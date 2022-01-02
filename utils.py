import cv2
import numpy as np
import matplotlib.pyplot as plt


def imageprep(screen_capture):
    '''
    This function should be run first
    :param screen_capture:
    :return: image w/o color border, dot count plus removal from image, arrow removal
    '''

    img = screen_capture

    # crop out outside color border
    top_left = (24, 24)
    bottom_right = (2413, 1102)
    img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

    # arrow-removal
    top_left_arrow = (35, 957)
    bottom_right_arrow = (200, 1049)

    donor = cv2.flip(img, 1)
    donor = donor[top_left_arrow[1]:bottom_right_arrow[1], top_left_arrow[0]:bottom_right_arrow[0], :]

    # this is to match the background gradient (picked by eye)
    donor += 4

    img[top_left_arrow[1]:bottom_right_arrow[1], top_left_arrow[0]:bottom_right_arrow[0], :] = donor

    # bottom trace to determine available dots
    trace = img[1025, np.arange(1000, 1350, 1), 0]
    half_jump = (np.max(trace) - np.min(trace)) / 2

    previous_point = trace[0]
    crosses = 0
    for point in trace[1::]:
        if abs(float(point) - float(previous_point)) > half_jump:
            crosses += 1
        previous_point = point

    dot_count = int(crosses / 2)

    # now remove dot(s)
    rows, cols = img.shape[0:2]
    donor = img[1050:rows, 1000:1350, :]

    img[999:(999 + donor.shape[0]), 1000:1350, :] = donor + 4
    img[1025:(1025 + donor.shape[0]), 1000:1350, :] = donor + 2

    return img, dot_count


if __name__ == '__main__':

    # read in the raw, screen-capture image
    #img = cv2.imread('IMG_1157.PNG')
    img = cv2.imread('image0.png')

    img, dot_count = imageprep(img)

    plt.imshow(img)
    plt.title(f'player dots = {dot_count}')
    plt.show()

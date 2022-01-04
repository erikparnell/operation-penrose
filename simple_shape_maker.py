import cv2
import numpy as np
import matplotlib.pyplot as plt


def color_pixel_count(img):

    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    count = np.count_nonzero(tmp)

    print(f'color pixel count = {count}')


if __name__ == '__main__':

    img = np.zeros((512, 512, 3), np.uint8)

    img_circle = cv2.circle(img.copy(), (255, 255), radius=20, color=(255, 0, 0), thickness=1)
    img_square = cv2.rectangle(img.copy(), (210, 210), (240, 240), color=(255, 0, 0), thickness=1)

    color_pixel_count(img_circle)
    color_pixel_count(img_square)

    cv2.imwrite('circle.png', cv2.cvtColor(img_circle, cv2.COLOR_RGB2BGR))
    cv2.imwrite('square.png', cv2.cvtColor(img_square, cv2.COLOR_RGB2BGR))

    plt.imshow(img_square)
    plt.show()

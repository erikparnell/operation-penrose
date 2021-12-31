import cv2
import fast_utils  # run 'compile_pyxes.py' at least once, first, to use this
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def remove_skeletal_spikes(skeleton):
    # this function is slow. fast_utils contains a C version

    tmp = skeleton.copy()
    ii = np.argwhere(tmp > 0)

    for i in ii:
        x = i[0]
        y = i[1]
        count = 0
        if tmp[x-1, y-1, 1] == 255:
            count += 1
        if tmp[x, y-1, 1] == 255:
            count += 1
        if tmp[x+1, y-1, 1] == 255:
            count += 1
        if tmp[x+1, y, 1] == 255:
            count += 1
        if tmp[x+1, y+1, 1] == 255:
            count += 1
        if tmp[x, y+1, 1] == 255:
            count += 1
        if tmp[x-1, y+1, 1] == 255:
            count += 1
        if tmp[x-1, y, 1] == 255:
            count += 1

        if count == 1:
            tmp[x, y, 0] = 0
            tmp[x, y, 1] = 0
            tmp[x, y, 2] = 0
            tmp = remove_skeletal_spikes(tmp)

    return tmp


def draw_skeleton(img, skeleton):
    # this function is also slow. Use fast_util's C implementation

    rows, cols = skelly.shape[0:2]

    for r in range(rows):
        for c in range(cols):
            if sum(skelly[r, c, :]) > 0:
                img[r, c, 0] = 255
                img[r, c, 1] = 0
                img[r, c, 2] = 0

    return img


if __name__ == '__main__':

    # 1) crop image 2) canny edge-detect 3) hand-paint-bucket fill track as white
    img = cv2.imread('hand_edits3.png')

    # track needs to be white
    img = 255 - img

    skelly = skeletonize(img)

    # skelly = remove_skeletal_spikes(skelly)
    #skelly = fast_utils.remove_skeletal_spikes(skelly)
    skelly = fast_utils.remove_skeletal_spikes(skelly, img)

    #img = draw_skeleton(img, skelly)
    img = np.array(fast_utils.draw_skeleton(img, skelly))

    plt.figure(1)
    plt.imshow(skelly)

    plt.figure(2)
    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.show()

    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('skeleton.png', tmp)

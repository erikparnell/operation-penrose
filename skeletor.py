import cv2
import fast_utils  # run 'compile_pyxes.py' at least once, first, before using
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def remove_skeletal_spikes(skeleton):
    # this function is slow. fast_utils contains a C version that is faster and more advanced

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


def get_skelly(bin_img):

    color_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    skelly = skeletonize(color_img)
    skelly = fast_utils.remove_skeletal_spikes(skelly, color_img)
    color_img = np.array(fast_utils.draw_skeleton(color_img, skelly))

    return color_img


if __name__ == '__main__':

    # hand-edits = manually...1) crop image 2) canny edge-detect 3) hand-paint-bucket fill track as white
    img = cv2.imread('hand_edits2.png')

    # track needs to be white
    img = 255 - img

    skelly = skeletonize(img)

    skelly = remove_skeletal_spikes(skelly)
    #skelly = fast_utils.remove_skeletal_spikes(skelly, img) # cython version

    # these will be the raw coordinates of the skeleton
    skeletal_indices = np.argwhere(skelly[:, :, 1] > 0)

    np.save('skelly_points.npy', skeletal_indices)

    # how to load the skeletal points from disk if you want to bypass calculating skeleton each run
    loaded_skeletal_indices = np.load('skelly_points.npy')

    img = draw_skeleton(img, skelly)
    #img = np.array(fast_utils.draw_skeleton(img, skelly)) # cython version

    #plt.figure(1)
    #plt.imshow(skelly)

    #plt.figure(2)
    #plt.imshow(img, 'gray')
    #plt.axis('off')
    #plt.show()

    #tmp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('skeleton.png', tmp)

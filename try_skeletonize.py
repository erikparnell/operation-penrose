import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d


if __name__ == '__main__':

    # 1) crop image 2) canny edge-detect 3) hand-paint-bucket fill track as white
    img = cv2.imread('hand_edits2.png')

    # track needs to be white
    img = 255 - img

    skelly = skeletonize(img)

    rows, cols = skelly.shape[0:2]

    # this'll be horribly slow in python, but is just to show if skeletonize worked ok
    for r in range(rows):
        for c in range(cols):
            if sum(skelly[r, c, :]) > 0:
                img[r, c, 0] = 0
                img[r, c, 1] = 0
                img[r, c, 2] = 255

    cv2.imwrite('skeleton.png', img)
    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.show()

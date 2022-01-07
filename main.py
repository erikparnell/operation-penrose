import cv2
import utils
import random
import skeletor
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    img = cv2.imread('IMG_1157.PNG') # Penrose level
    #img = cv2.imread('image_1158.PNG')  # Penrose level
    #img = cv2.imread('image.png') # Penrose level
    #img = cv2.imread('image0.png')
    #img = cv2.imread('image1.png')
    #img = cv2.imread('image2.png')  # for this one, you'll need --> remove_spikes=False

    img, player_dots, enemies = utils.prepare_image(img)
    skelly = skeletor.get_skelly(img, remove_spikes=True)  # set False for any stage that looks like spaghetti

    #ii_red = np.argwhere((skelly[:, :, 0] == 255)*(skelly[:, :, 1] == 0))
    #ii_red = sorted(ii_red, key=lambda p: (p[0], p[1]))
    #random.shuffle(ii_red)

    #for (q, i) in enumerate(ii_red):
    #    if q % 5 == 0:
    #        cv2.circle(skelly, (i[1], i[0]), 5, (0, 0, 255), 1)

    cv2.imwrite('skelly.png', skelly)
    np.save('enemies.npy', enemies, allow_pickle=True)

    plt.imshow(skelly)
    plt.show()

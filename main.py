import cv2
import utils
import solver
import skeletor
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #img = cv2.imread('IMG_1157.PNG')  # Penrose level
    #img = cv2.imread('image_1158.png')  # Penrose level
    img = cv2.imread('image.png')  # Penrose level
    #img = cv2.imread('image0.png')
    #img = cv2.imread('image1.png')
    #img = cv2.imread('image2.png')  # for this one, you'll need --> remove_spikes=False

    img_orig = img.copy()

    img, player_dots, enemies = utils.prepare_image(img, image_scale_factor=0.5)
    skelly = skeletor.get_skelly(img, remove_spikes=True)  # set False for any stage that looks like spaghetti

    cv2.imwrite('skelly.png', skelly)
    np.save('enemies.npy', enemies, allow_pickle=True)

    # right now solver only works for a single player dot
    sol_img = solver.solve(skelly, enemies)  # you'll first need to run once compile_pyxes.py again

    plt.figure(1)
    plt.imshow(img_orig)

    plt.figure(2)
    plt.imshow(sol_img)
    plt.show()

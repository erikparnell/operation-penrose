import cv2
import utils
import skeletor
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #img = cv2.imread('IMG_1157.PNG')
    #img = cv2.imread('image0.png')
    img = cv2.imread('image1.png')

    img, player_dots, enemies = utils.prepare_image(img, do_plot=False)
    skelly = skeletor.get_skelly(img)

    plt.imshow(skelly)
    plt.show()

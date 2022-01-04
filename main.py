import cv2
import utils
import skeletor
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #img = cv2.imread('IMG_1157.PNG')
    #img = cv2.imread('image0.png')
    img = cv2.imread('image1.png')
    #img = cv2.imread('image2.png')  # for this one, you'll need --> remove_spikes=False

    img, player_dots, enemies = utils.prepare_image(img)
    skelly = skeletor.get_skelly(img, remove_spikes=True)  # set False for any stage that looks like spaghetti

    plt.imshow(skelly)
    plt.show()

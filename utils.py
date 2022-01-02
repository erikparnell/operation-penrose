import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class Enemy:

    def __init__(self, color, location):
        self.color = color
        self.location = location
        self.score = 0


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


def findenemies(img):
    '''
    first run imageprep to ensure color border is removed
    :param img:
    :return:
    '''

    def isgray(pixel):
        return pixel[0] == pixel[1] == pixel[2]

    # we know the enemies are colored pixels
    R = img[:, :, 0].astype(int)
    G = img[:, :, 1].astype(int)
    B = img[:, :, 2].astype(int)

    M = (R + G + B) / 3
    M = M.astype(int)

    # pixel coordinates of all enemies
    ii_enemies = np.argwhere(M != R)

    # return now if ii_enemies is empty
    if not ii_enemies.any():
        return [], []

    # encircle and gray out first enemy cluster you find
    i = ii_enemies[0]

    # walk left until you reach an all-gray pixel
    while True:
        i[0] -= 1
        if isgray(img[i[0], i[1]]):
            break

    # walk back right once, back into color
    i[0] += 1

    darkest_gray = 255

    # initial director. Known, since we chose to initially walk left
    director = -1 + 0j

    enemy_contour = []

    keep_going = True
    while keep_going:
        for t in np.arange(0, 360, 45):
            vec = director*np.exp(1j*t*np.pi/180)
            x = int(np.round(vec.real))
            y = int(np.round(vec.imag))
            pixel = img[i[0]-y, i[1]+x]
            if isgray(pixel):
                if pixel[0] < darkest_gray:
                    darkest_gray = pixel[0]
            else:
                # update director and i
                vec *= -1
                vec *= np.exp(1j*np.pi/4)
                director = vec
                #print(f'({i[1]+x}, {i[0]-y})')  # for debug
                #img[i[0]-y, i[1]+x, :] = [0, 255, 0]  # for debug
                enemy_contour.append([i[0]-y, i[1]+x])
                i = [i[0]-y, i[1]+x]
                if i[0] == ii_enemies[0][0] and i[1] == ii_enemies[0][1]:
                    keep_going = False
                break

    # convert enemy_contours into opencv-compatible object
    enemy_contour = np.array([[[p[1], p[0]]] for p in enemy_contour]).astype(np.int32)

    # get all pixel coords of this enemy
    mask = np.zeros(img.shape[0:2], np.uint8)
    cv2.drawContours(mask, [enemy_contour], 0, 255, -1)
    pixel_coords = np.transpose(np.nonzero(mask))

    # determine this enemy's color
    xx = [coord[0] for coord in pixel_coords]
    yy = [coord[1] for coord in pixel_coords]
    sub_img = img[xx, yy, :]

    reds = []
    greens = []
    blues = []
    for p in sub_img:
        reds.append(p[0])
        greens.append(p[1])
        blues.append(p[2])

    red = stats.mode(reds)[0][0]
    green = stats.mode(greens)[0][0]
    blue = stats.mode(blues)[0][0]

    enemy = Enemy([red, green, blue], [int(round(np.mean(xx))), int(round(np.mean(yy)))])

    # now that we have found an enemy...remove it
    cv2.drawContours(img, [enemy_contour], -1, (int(darkest_gray), int(darkest_gray), int(darkest_gray)), -1)

    return img, enemy


if __name__ == '__main__':

    # read in the raw, screen-capture image
    img = cv2.imread('IMG_1157.PNG')
    #img = cv2.imread('image0.png')

    img_orig = img.copy()

    img, dot_count = imageprep(img)

    out_img = img.copy()
    enemies = []
    while True:
        img, enemy = findenemies(img)
        if enemy:
            enemies.append(enemy)
            out_img = img.copy()
        else:
            break

    print(f'\nplayer dots: {dot_count}')
    print('list of enemies:')
    for (q, enemy) in enumerate(enemies):
        print(f'\tEnemy {q+1} has RGB color ({enemy.color[0]}, {enemy.color[1]}, {enemy.color[2]})'
              f' at prepared position ({enemy.location[1]}, {enemy.location[0]})')

    plt.figure(1)
    plt.imshow(out_img)
    plt.title('prepared stage')

    plt.figure(2)
    plt.imshow(img_orig)
    plt.title('raw stage')

    plt.show()

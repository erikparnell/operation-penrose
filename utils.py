import cv2
import fast_utils
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class Enemy:

    def __init__(self, color, location):
        self.color = color
        self.location = location
        self.score = 0


def count_dots(img):

    # bottom trace to determine available dots
    trace = img[1025, np.arange(1000, 1350, 1), 0].astype(float)

    midline = (np.max(trace) + np.min(trace)) / 2
    midline = np.floor(midline) + 0.5  # ensure it's a non-integer

    crosses = 0
    previous_point = trace[0]
    for point in trace[1::]:
        if (previous_point - midline)*(point - midline) < 0:
            crosses += 1
        previous_point = point

    dot_count = int(crosses / 2)

    return dot_count


def imageprep1(screen_capture):
    '''
    level-1 image preparation. This function should be run first
    :param screen_capture:
    :return: image w/o color border, dot count plus removal from image, arrow removal
    '''

    img = screen_capture

    # crop out outside color border
    top_left = (24, 24)
    bottom_right = (2413-1, 1102-1)
    img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

    # arrow-removal
    top_left_arrow = (35, 957)
    bottom_right_arrow = (200, 1049)

    donor = cv2.flip(img, 1)
    donor = donor[top_left_arrow[1]:bottom_right_arrow[1], top_left_arrow[0]:bottom_right_arrow[0], :]

    # this is to match the background gradient (picked by eye)
    donor += 4

    img[top_left_arrow[1]:bottom_right_arrow[1], top_left_arrow[0]:bottom_right_arrow[0], :] = donor

    dot_count = count_dots(img)

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
    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 2].astype(float)

    delta1 = np.abs(R - G)
    delta2 = np.abs(R - B)
    delta3 = np.abs(G - B)

    delta = delta1 + delta2 + delta3

    # pixel coordinates of all enemies
    ii_enemies = np.argwhere(delta > 1)

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


def remove_background_gradient(gray_img):

    rows, cols = gray_img.shape

    sim_background = np.zeros((rows, cols))

    top_trace = gray_img[0, :]
    right_trace = gray_img[:, -1]

    xx_top = range(cols)
    xx_right = range(rows)

    pfit_top = np.polyfit(xx_top, top_trace, 2)
    pfit_right = np.polyfit(xx_right, right_trace, 2)

    fast_utils.gradient_helper(sim_background, pfit_top[0], pfit_top[1], pfit_top[2], pfit_right[0], pfit_right[1])
    #print(sim_background[0, 1:10])
    #for r in range(rows):
    #    for c in range(cols):
    #        offset = np.polyval(pfit_top, c)
    #        sim_background[r, c] = np.polyval([pfit_right[0], pfit_right[1], offset], r)

    ave = np.mean(sim_background)
    sim_background -= ave
    sim_background *= -1
    sim_background += ave

    sim_background = np.round(sim_background).astype(np.int32)
    gray_img = gray_img.astype(np.int32) + sim_background
    gray_img = 0.5*gray_img
    gray_img = np.round(gray_img).astype(np.uint8)

    return gray_img


def imageprep2(img):
    # level-2 preparation of image

    tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp_img = remove_background_gradient(tmp_img)
    otsu_threshold, otsu_img = cv2.threshold(tmp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_out = cv2.bitwise_not(otsu_img)


    '''
    # canny edge left gaps that didn't work well with floodfill
    edges = cv2.Canny(tmp_img, 30, 70)  # TODO remove hard-coded values?

    # find a single point inside the track as the seed-point for flood fill
    h, w = edges.shape[0:2]

    i_white = np.argwhere(edges[int(h/2), :] > 0).flatten()
    ia = i_white[0]
    for ib in i_white[1:]:
        if abs(ib-ia) > 1:
            x = int(round((ia + ib)/2))
            break
        ia = ib

    seed_point = (x, int(h/2))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(edges, mask, seed_point, 255, flags=4 | (255 << 8))

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.circle(edges, (x, int(h/2)), 3, (255, 0, 0), -1)

    return edges
    '''

    return img_out


def prepare_image(img, do_plot = False):

    img_orig = img.copy()

    img_plevel1, player_dots = imageprep1(img)

    out_img = img_plevel1.copy()
    enemies = []
    while True:
        img_plevel1, enemy = findenemies(img_plevel1)
        if enemy:
            enemies.append(enemy)
            out_img = img_plevel1.copy()
        else:
            break

    # so, the background gradient likely slightly tweaks the true enemy color
    # and the removal of the gradient isn't going to be perfect, so some re-adjustment
    # of the enemy colors is necessary
    for (q, enemy) in enumerate(enemies):
        other_enemies = [enemies[i] for i in np.setdiff1d(np.arange(0, len(enemies), 1), [q])]
        for other_enemy in other_enemies:
            tmp = np.abs(np.array(enemy.color).astype(float) - np.array(other_enemy.color).astype(float))
            if np.max(tmp) <= 15:
                other_enemy.color = enemy.color

    print(f'\nplayer dots: {player_dots}')
    print('list of enemies:')
    for (q, enemy) in enumerate(enemies):
        print(f'\tEnemy {q + 1} has RGB color ({enemy.color[0]}, {enemy.color[1]}, {enemy.color[2]})'
              f' at prepared position ({enemy.location[1]}, {enemy.location[0]})')

    img_plevel2 = imageprep2(out_img)

    if do_plot:
        plt.figure(1)
        plt.imshow(out_img)
        plt.title('prepared stage')

        plt.figure(2)
        plt.imshow(img_orig)
        plt.title('raw stage')

        plt.figure(3)
        plt.imshow(img_plevel2, 'gray')

        plt.show()

    return img_plevel2, player_dots, enemies


if __name__ == '__main__':

    # read in the raw, screen-capture image
    img = cv2.imread('IMG_1157.PNG')
    #img = cv2.imread('image0.png')

    prepare_image(img)

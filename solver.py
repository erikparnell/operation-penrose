import cv2
import time
import fast_utils
import numpy as np
import multiprocessing
from utils import Enemy
from copy import deepcopy
import matplotlib.pyplot as plt

unclaimed_track_color = (200, 200, 200)


def calculate_scores(img, players):
    # this won't work for collaborative enemies/enemies of the same color

    for player in players:
        b, g, r = player.color
        player.score = np.count_nonzero((img[:, :, 0] == r) * (img[:, :, 1] == g) * (img[:, :, 2] == b))

    return players


def _solve(skelly, enemies, user_location, player_color_matrix, active_points_matrix_x, active_points_matrix_y, do_plot=False):

    img = skelly.copy()
    user = Enemy([0, 255, 0], user_location)
    players = list(enemies)
    players.append(user)

    # player placement
    for player in players:
        color = (int(player.color[2]), int(player.color[1]), int(player.color[0]))
        img[player.location[0], player.location[1], :] = color

    slo_mo = False

    if slo_mo:
        keep_going = True
        while keep_going:
            keep_going = False
            for player in players:
                new_active_points = []
                for active_point in player.active_points:
                    keep_going = True
                    for (dx, dy) in zip([1, 1, 0, -1, -1, -1, 0, 1], [0, 1, 1, 1, 0, -1, -1, -1]):
                        pixel = img[active_point[0] + dx, active_point[1] + dy, :]
                        if pixel[0] == unclaimed_track_color[0] and pixel[1] == unclaimed_track_color[1] and pixel[2] == \
                                unclaimed_track_color[2]:
                            new_active_points.append([active_point[0] + dx, active_point[1] + dy])
                            img[active_point[0] + dx, active_point[1] + dy, :] = [player.color[2], player.color[1],
                                                                                  player.color[0]]
                player.active_points = new_active_points
    else:

        # TODO do i really need to re-zero these?
        player_color_matrix = 0*player_color_matrix
        active_points_matrix_x = 0*active_points_matrix_x
        active_points_matrix_y = 0*active_points_matrix_y
        for (q, player) in enumerate(players):
            player_color_matrix[q, :] = player.color[::-1]
            active_points_matrix_x[0, q] = player.active_points[0][0]
            active_points_matrix_y[0, q] = player.active_points[0][1]

        fast_utils.solver_helper(img, player_color_matrix, active_points_matrix_x, active_points_matrix_y,
                                 0 * active_points_matrix_x, 0 * active_points_matrix_y)

    if do_plot:
        for player in players:
            b, g, r = player.color
            ii_player = np.argwhere((img[:, :, 0] == r) * (img[:, :, 1] == g) * (img[:, :, 2] == b))
            for i in ii_player:
                cv2.circle(img, (i[1], i[0]), 5, (int(r), int(g), int(b)), -1)

        plt.imshow(img)
        plt.show()

    calculate_scores(img, players)
    max_score = -1
    winning_player = -1
    for (q, player) in enumerate(players):
        if player.score > max_score:
            max_score = player.score
            winning_player = q

    # the user has won
    if winning_player == (len(players)-1):
        return True, user_location
    else:
        return False, []


def run_solver(skelly, enemies, user_spots, do_print=False):

    # some better objects for C/cython to use
    player_color_matrix = np.zeros((len(enemies)+1, 3), np.uint8)
    active_points_matrix_x = np.zeros((100, len(enemies)+1), np.uint16)
    active_points_matrix_y = np.zeros((100, len(enemies)+1), np.uint16)

    interval = 1
    winning_spots = []
    for (q, user_spot) in enumerate(user_spots):
        if do_print:
            percent_complete = 100*q/len(user_spots)
            if percent_complete > interval:
                print('{:0.0f}% complete'.format(percent_complete))
                interval += 1
        did_user_win, winning_location = _solve(skelly, deepcopy(enemies), user_spot, player_color_matrix,
                                                active_points_matrix_x, active_points_matrix_y, do_plot=False)
        if did_user_win:
            winning_spots.append(winning_location)

    return winning_spots


def solve(skelly, enemies):

    use_mp = True

    # locations of track
    ii_track = np.argwhere((skelly[:, :, 0] == 255) * (skelly[:, :, 1] == 0))

    # fix to make sure enemies are on track
    for enemy in enemies:
        delta = [(enemy.location[0] - pt[0]) ** 2 + (enemy.location[1] - pt[1]) ** 2 for pt in ii_track]
        i_min = np.argmin(delta)
        min_location = ii_track[i_min]
        enemy.location = min_location
        enemy.active_points = [min_location]

    # find points that user is allowed to play (i.e. not atop enemy points)
    user_spots = []
    enemy_spots = np.array([e.location for e in enemies])
    for i in ii_track:
        if i not in enemy_spots:
            user_spots.append(i)

    # user_spots = [[309, 1459]] # for debug

    # i want to change the skeleton/track to a light gray color
    h, w = skelly.shape[0:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(skelly, mask, (ii_track[0][1], ii_track[0][0]), unclaimed_track_color, flags=8 | (255 << 8))

    now = time.time()

    if use_mp:

        # chop up data for multiprocessing
        cpus = multiprocessing.cpu_count()
        p_enemies = [enemies for i in range(cpus)]
        p_skelly = [skelly.copy() for i in range(cpus)]
        p_user_spots = []
        for k in range(cpus):
            chunk_size = int(np.floor(len(user_spots) / cpus))
            i_start = k * chunk_size
            i_stop = (k + 1) * chunk_size

            if k == cpus:
                i_stop = len(user_spots) - 1

            p_user_spots.append(user_spots[i_start:i_stop])

        pool = multiprocessing.Pool(processes=cpus)
        async_results = [pool.apply_async(run_solver, [p_skelly[r], p_enemies[r], p_user_spots[r]], ) for r in
                         range(cpus)]
        all_results = [result.get() for result in async_results]
        winning_spots = [item for sublist in all_results for item in sublist]
    else:
        winning_spots = run_solver(skelly, enemies, user_spots, do_print=True)

    then = time.time()

    print('run time: {:0.2f} seconds'.format(then - now))

    for winning_spot in winning_spots:
        cv2.circle(skelly, (winning_spot[1], winning_spot[0]), 5, (255, 0, 0), -1)

    return skelly


if __name__ == '__main__':

    # inputs
    skelly = cv2.imread('skelly.png')
    enemies = np.load('enemies.npy', allow_pickle=True)

    sol_img = solve(skelly, enemies)

    plt.imshow(sol_img)
    plt.show()



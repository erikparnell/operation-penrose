import cv2
import numpy as np
from utils import Enemy
import matplotlib.pyplot as plt


if __name__ == '__main__':

    skelly = cv2.imread('skelly.png')
    enemies = np.load('enemies.npy', allow_pickle=True)
    img = skelly.copy()

    # locations of track
    ii_track = np.argwhere((img[:, :, 0] == 255) * (skelly[:, :, 1] == 0))

    # fix to make sure enemies are on track, as enemy-blob's average may not necessarily fall upon skeleton track
    for enemy in enemies:
        delta = [(enemy.location[0] - pt[0]) ** 2 + (enemy.location[1] - pt[1]) ** 2 for pt in ii_track]
        i_min = np.argmin(delta)
        min_location = ii_track[i_min]
        enemy.location = min_location
        enemy.active_points = [min_location]

    # i want to make the skeleton a light gray instead of red
    unclaimed_track_color = (200, 200, 200)
    h, w = img.shape[0:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img, mask, (ii_track[0][1], ii_track[0][0]), unclaimed_track_color, flags=8 | (255 << 8))

    # the user is technically a player
    user = Enemy([0, 255, 0], location=[626, 1195])
    #user = Enemy([0, 255, 0], location=[490, 1202])
    #user = Enemy([0, 255, 0], location=[96, 1268])
    players = list(enemies)
    players.append(user)

    # player placement
    for player in players:
        color = (int(player.color[2]), int(player.color[1]), int(player.color[0]))
        img[player.location[0], player.location[1], :] = color

    height, width = skelly.shape[0:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # works and file size is much smaller
    fps = 180
    img_for_vid = img.copy()
    vid_out = cv2.VideoWriter('video.avi', fourcc, fps, (width, height))
    #vid_out = cv2.VideoWriter('video.avi', fourcc, fps, (int(width/2), int(height/2))) # doesn't seem to work

    # play out this setup
    keep_going = True
    while keep_going:
        vid_out.write(img_for_vid)
        keep_going = False
        for player in players:
            r, g, b = player.color
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
                        player.captured_points.append([active_point[0] + dx, active_point[1] + dy])
                        cv2.circle(img_for_vid, (active_point[1] + dy, active_point[0] + dx), 5, (int(r), int(g), int(b)), -1)

            player.active_points = new_active_points

    vid_out.release()

    plt.imshow(img)
    plt.show()

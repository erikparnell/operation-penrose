import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def find_nearest(points_list, current_point):
    """Return a list of the nearest point/s."""
    closest = []
    min_dist = None
    for entry in points_list:
        if entry["owner"] is None:
            new_dist = math.dist(entry["coords"], current_point)
            if min_dist is None or new_dist == min_dist:
                closest.append(entry["point_num"])
                min_dist = new_dist
            elif new_dist < min_dist:
                min_dist = new_dist
                closest[-1] = entry["point_num"]
            else:
                pass
    return closest


def find_nearest(points_list, current_point):
    """Return a list of the nearest point/s."""
    closest = []
    min_dist = None
    for entry in points_list:
        if entry["owner"] is None:
            new_dist = math.dist(entry["coords"], current_point)
            if min_dist is None or new_dist == min_dist:
                closest.append(entry["point_num"])
                min_dist = new_dist
            elif new_dist < min_dist:
                min_dist = new_dist
                closest[-1] = entry["point_num"]
            else:
                pass
    return closest


def all_owned(current_list):
    for entry in current_list:
        if entry["owner"] is None:
            return False
    return True


def simulate(points_list, **starting_points):
    """Returns the final points list based on start coordinates"""
    updated_list = points_list #initialize updated list

    head_points = {} #mapping of players to current head point/s
    for key, value in starting_points.items(): #initalize scores
        head_points[key] = []

    for key, value in starting_points.items(): #assign first round ownership
        nearest_point = find_nearest(points_list, value)
        for point in points_list:
            if point["point_num"] == nearest_point[0]:
                point["owner"] = key
        head_points[key] = nearest_point #update head points

    counter = 0
    while all_owned(updated_list) is False:
        #take ownership of neighbors
        for player in head_points.items():
            player_name = player[0]
            for head_point_num in player[1]:
                for point in updated_list:
                    if point["point_num"] == head_point_num:
                        neighbors = point["neighbors"]
                        for neighbor in neighbors:
                            for point in updated_list:
                                if point["point_num"] == neighbor and point["owner"] is None:
                                    point["owner"] = player_name
                                    print(f'counter = {counter}')
                                    counter += 1
                        head_points[player_name] = [neighbor]
    return updated_list


def calc_scores(final_list, **player_names):
    scores = player_names
    for point in final_list:
        if point["owner"] in player_names:
            scores[point["owner"]] = scores[point["owner"]] + 1
    return scores


def main():

    user_start = [280, 220]
    red_start = [401, 175]
    blue_start = [401, 188]

    skelly = cv2.imread('testbed.png')

    a = np.argwhere(skelly[:, :, 2] > 0)

    # b = [{'point_num': (n+1), 'coords': a[n], 'owner': None, 'neighbor': []} for n in range(len(a))]
    b = [{'point_num': (n + 1), 'coords': [a[n][0], a[n][1]], 'owner': None, 'neighbors': []} for n in range(len(a))]

    # a = np.array([[1, 1], [2, 2], [3, 3]])

    for (k, coord) in enumerate(a):
        ii = np.setdiff1d(np.arange(0, len(a), 1), [k])
        other_coods = a[ii]
        distances = np.array([math.dist(coord, coordk) for coordk in other_coods])
        i_neighbors = np.argwhere(distances < 1.4143).flatten()
        b[k]['neighbors'] = list(ii[i_neighbors])

    points_list = b

    final_list = simulate(points_list, user=user_start, red=red_start, blue=blue_start)
    for each in final_list:
        print(each)
    scores = calc_scores(final_list, blue=0, red=0, user=0)
    print(scores)


if __name__ == '__main__':

    main()


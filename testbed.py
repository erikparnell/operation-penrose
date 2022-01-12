import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import time


def find_nearest(points_list, current_point):
    """Returns a list of the nearest point/s"""
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
    return [closest[0]]


def all_owned(current_list):
    """Checks to see if all points are owned"""
    for entry in current_list:
        if entry["owner"] is None:
            return False
    return True


def simulate(points_list, **starting_points):
    """Returns the final points list based on start coordinates"""
    updated_list = copy.deepcopy(points_list) #initialize updated list

    #none_count = False
    owned = False
    for point in updated_list:
        if point["owner"] is not None:
            owned = True
            break

    if not owned:
        head_points = {} #mapping of players to current head point/s
        for key, value in starting_points.items(): #initalize headpoints
            head_points[key] = []

        for key, value in starting_points.items(): #assign first round ownership
            nearest_point = find_nearest(points_list, value)
            for point in updated_list:
                if point["point_num"] == nearest_point[0]:
                    point["owner"] = key #assign owner
                    break
            head_points[key] = nearest_point #update head points

    while all_owned(updated_list) is False:
    #while True:
        for player in head_points.items():
            player_name = player[0]
            for head_point_num in player[1]:
                for point in updated_list:
                    if point["point_num"] == head_point_num:
                        for neighbor in point["neighbors"]:
                            for point in updated_list:
                                if point["owner"] is None and point["point_num"] == neighbor:
                                    point["owner"] = player_name #take ownership of point
                                    owned += 1
                                    head_points[player_name].append(point["point_num"]) #add newly owned point to head points                                   
                head_points[player_name].remove(head_point_num) #remove old head points
        #print(none_count)
        #if none_count <= 0:
            #break
    return updated_list


def calc_scores(final_list, **player_names):
    """Totals the point ownership and returns the score"""
    scores = player_names
    for point in final_list:
        if point["owner"] in player_names:
            scores[point["owner"]] = scores[point["owner"]] + 1
    return scores


def create_points_list(a):
    """Converts skeleton list to templated dict including neighboring points"""
    points_list = [{'point_num': (n + 1), 'coords': [a[n][0], a[n][1]], 'owner': None, 'neighbors': []} for n in range(len(a))]
    max = math.sqrt(2) + .0001
    for point in points_list:
        neighbors = []
        coords = point["coords"]
        for oth_point in points_list:
            oth_coords = oth_point["coords"]
            dist = math.dist(coords, oth_coords)
            if 0 < dist <= max:
                neighbors.append(oth_point["point_num"])
        point['neighbors'] = neighbors
    return points_list


def main():

    img = 'testbed.png'
    skelly = cv2.imread(img)

    a = np.argwhere(skelly[:, :, 2] > 0)

    coords_list = a.tolist()
    points_list = create_points_list(coords_list)

    #start = timeit.timeit()
    #final_list = simulate(points_list, user=user_start, red=red_start, blue=blue_start)
    #end = timeit.timeit()
    #print(f'It took {elapsed} seconds to complete the simulate function on {img}.')

    #scores = calc_scores(final_list, blue=0, red=0, user=0)
    #print(scores)
    #for key, value in scores.items():
        #sum = sum + value
    #print(f'Total points = {sum}')
    
    #user_start = [10, 210]
    red_start = [40, 175]
    blue_start = [20, 240]

    #brute force algorithm
    max_score = 0
    max_score_coords = []
    counter = 0
    now = time.time()
    for point in points_list:
        counter += 1
        print(counter)
        if point["coords"] != red_start or blue_start: #can i do 'or' statement like this?
            user_coords = point["coords"]
            final_list = simulate(points_list, user=user_coords, red=red_start, blue=blue_start)
            scores = calc_scores(final_list, blue=0, red=0, user=0)
            user_score = scores["user"]
            if user_score > max_score:
                max_score = user_score
                max_score_coords = user_coords
            
        #need to clear out ownership point next round of simulate (trying with copy in simulator)
    then = time.time()
    elapsed = then - now
    print(f'Best coordinates to choose are {max_score_coords} resulting in a score of {max_score} and it took {elapsed} seconds')


def dantooine():
    # Dan's little world of some light mods to the code in def main

    # data created and saved in main.py
    skelly = cv2.imread('skelly.png')
    enemies = np.load('enemies.npy', allow_pickle=True)

    ii_track = np.argwhere((skelly[:, :, 0] == 255) * (skelly[:, :, 1] == 0))
    points_list = [{'point_num': (n + 1), 'coords': [ii_track[n][0], ii_track[n][1]], 'owner': None,
                    'neighbors': []} for n in range(len(ii_track))]

    # populate neighbors by using skelly as a lookup table
    neighbors = [[0, 0]]*8  # you'll never have more than 8 neighbors
    for point in points_list:
        neighbor_count = 0
        for (dx, dy) in zip([1, 1, 0, -1, -1, -1, 0, 1], [0, 1, 1, 1, 0, -1, -1, -1]):
            x, y = point['coords']
            pixel = skelly[x + dx, y + dy, :]
            if pixel[0] == 255 and pixel[1] == 0 and pixel[2] == 0:
                neighbors[neighbor_count] = [x + dx, y + dy]
                neighbor_count += 1
        point['neighbors'] = neighbors[0:neighbor_count]


    dbg = 1


if __name__ == '__main__':

    #main()
    dantooine()


    # b = [{'point_num': (n+1), 'coords': a[n], 'owner': None, 'neighbor': []} for n in range(len(a))]
    #b = [{'point_num': (n + 1), 'coords': [a[n][0], a[n][1]], 'owner': None, 'neighbors': []} for n in range(len(a))]

    # a = np.array([[1, 1], [2, 2], [3, 3]])

    #for (k, coord) in enumerate(a):
        #ii = np.setdiff1d(np.arange(0, len(a), 1), [k])
        #other_coods = a[ii]
        #distances = np.array([math.dist(coord, coordk) for coordk in other_coods])
        #i_neighbors = np.argwhere(distances < 1.4143).flatten()
        #b[k]['neighbors'] = list(ii[i_neighbors])

    #print(type(a))
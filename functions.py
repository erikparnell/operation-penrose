import math

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
                        head_points[player_name] = [neighbor]                            
    return updated_list

def calc_scores(final_list, **player_names):
    scores = player_names
    for point in final_list:
        if point["owner"] in player_names:
            scores[point["owner"]] = scores[point["owner"]] + 1
    return scores

def main():

    points_list = [
        {"point_num" : 1, "coords" : [300, 300], "owner" : None, "neighbors" : [2]},
        {"point_num" : 2, "coords" : [310, 310], "owner" : None, "neighbors" : [1, 3]},
        {"point_num" : 3, "coords" : [320, 320], "owner" : None, "neighbors" : [2, 4]},
        {"point_num" : 4, "coords" : [330, 330], "owner" : None, "neighbors" : [3, 5]},
        {"point_num" : 5, "coords" : [340, 340], "owner" : None, "neighbors" : [4, 6]},
        {"point_num" : 6, "coords" : [350, 350], "owner" : None, "neighbors" : [5, 7]},
        {"point_num" : 7, "coords" : [360, 360], "owner" : None, "neighbors" : [6, 8]},
        {"point_num" : 8, "coords" : [370, 370], "owner" : None, "neighbors" : [7, 9]},
        {"point_num" : 9, "coords" : [380, 380], "owner" : None, "neighbors" : [8, 10]},
        {"point_num" : 10, "coords" : [390, 390], "owner" : None, "neighbors" : [9]},
    ]

    

    user_start = [334, 335]
    red_start = [299, 299]
    blue_start = [391, 391]

    final_list = simulate(points_list, user=user_start, red=red_start, blue=blue_start)
    for each in final_list:
        print(each)
    scores = calc_scores(final_list, blue=0, red=0, user=0)
    print(scores)

if __name__ == "__main__":
    main()
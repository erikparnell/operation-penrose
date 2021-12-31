import math

points_list = [
    {"point_num" : 1, "coords" : [300, 300], "owner" : "blue"},
    {"point_num" : 2, "coords" : [310, 310], "owner" : None},
    {"point_num" : 3, "coords" : [320, 320], "owner" : None},
    {"point_num" : 4, "coords" : [330, 330], "owner" : None},
    {"point_num" : 5, "coords" : [340, 340], "owner" : None},
    {"point_num" : 6, "coords" : [350, 350], "owner" : "red"}
]

current_point = [335, 335]

def nearest_point(points_list, current_point):
    """This function will return a list of the nearest point/s."""
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
        


def main():
    closest_points = nearest_point(points_list, current_point)
    print(closest_points)

if __name__ == "__main__":
    main()
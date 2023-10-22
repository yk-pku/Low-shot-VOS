import numpy as np
import math

# p->point:(x, y)
def point_dis(p1, p2):
    dis = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dis

def point_disturb(points, mask, args):
    aug_points = []
    h, w = mask.shape
    for i, point in enumerate(points):
        x, y = point
        x_offset = np.random.choice([-1, 0, 1], 1)[0]
        y_offset = np.random.choice([-1, 0, 1], 1)[0]
        
        x_aug = x + x_offset * (np.random.choice(args.point_max_off, 1)[0] + 1)
        y_aug = y + y_offset * (np.random.choice(args.point_max_off, 1)[0] + 1)
        
        x_aug = min(x_aug, w)
        x_aug = max(x_aug, 0)
        y_aug = min(y_aug, h)
        y_aug = max(y_aug, 0)
        
        aug_points.append((x_aug, y_aug))
        
    return np.array(aug_points)

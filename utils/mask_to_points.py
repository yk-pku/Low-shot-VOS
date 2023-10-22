import numpy as np
import torch
import random
import bisect
import cv2
import matplotlib.pyplot as plt
import os


out_path = 'test_learnpos_results/00ad5016a4-00040_contour'

def mask_to_points(mask, num = 30):
    y_coords, x_coords = np.where(mask > 0.5)
    points = []
    labels = []
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        pair_coord = [x, y]
        points.append(pair_coord)
        labels.append(1)

    points = np.array(points)
    labels = np.array(labels)
    if len(points) > num:
        points_indices = np.random.choice(len(points), size = num, replace= False)
        points = points[points_indices]
        labels = labels[:num]

    return points, labels

def mask_to_box(mask):
    y_coords, x_coords = np.where(mask > 0.5)
    points = list(zip(x_coords,y_coords))
        
    if len(points) == 0:
        return None

    left = np.min(x_coords); right = np.max(x_coords)
    low = np.min(y_coords);  high = np.max(y_coords)

    box = np.array([left,low,right,high])
    return  box

def mask_to_points_grid(mask, point_num = 10):
    y_coords, x_coords = np.where(mask > 0.5)
    points = list(zip(x_coords,y_coords))

    if len(points) == 0:
        h, w = mask.shape
        points = [(w/2, h/2)]
        return np.array(points), np.ones(len(points))

    left = np.min(x_coords); right = np.max(x_coords)
    low = np.min(y_coords);  high = np.max(y_coords)

    points_ret = []
    test_grid = [2,3,4,5,6,8,11,14]
    gap = 1000

    for num in test_grid:
        r_points = []
        x_interval = max((right - left + 1)//num,1)
        y_interval = max((high - low + 1)//num,1)

        for i in range(left ,right + 1 ,x_interval):
            for j in range(low  , high + 1 , y_interval):
                # points in region[i:i+x_interval, j:j+y_interval]
                y_coords_part, x_coords_part = np.where(mask[j:j+y_interval,i:i+x_interval] > 0.5)
                num_points_part = len(y_coords_part)

                # if points in region, choose 'center'
                if num_points_part:
                    x_coords_part += i
                    y_coords_part += j
                    y_part_mid = (y_coords_part[0] + y_coords_part[-1])//2
                    x_part_mid = (min(x_coords_part) + max(x_coords_part))//2
                    if mask[y_part_mid][x_part_mid]:
                        r_points.append((x_part_mid,y_part_mid))
                    else:
                        mid_idx = (num_points_part-1) // 2
                        r_points.append((x_coords_part[mid_idx],y_coords_part[mid_idx]))

        if abs(len(r_points)-point_num) < gap:
            gap = abs(len(r_points)-point_num)
            points_ret = r_points
            if gap == 0:
                break
    return np.array(points_ret), np.ones(len(points_ret))

# Polygon fitting coutour
def mask_to_points_coutours(mask):
    img = cv2.imread(mask)

    contour = draw_contour(img,(0,0,255),2)
    print(type(contour))
    epsilon = 0.001*cv2.arcLength(contour[0],True)
    approx = cv2.approxPolyDP(contour[0], epsilon, True)
    approx=np.squeeze(approx)
    print(type(approx),approx.shape)

    res = cv2.drawContours(img.copy(),[approx],-1,(0,0,255),2)
    plt.figure(figsize=(20,20))
    plt.imshow(res)
    plt.savefig(os.path.join(out_path, 'coutour1.png'))

    return np.array(approx), np.ones(len(approx))


#return contour
def draw_contour(img,color,width):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret , binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    contour,hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#find contour
    copy = img.copy()
    result = cv2.drawContours(copy,contour[:],-1,color,width)
    plt.figure(figsize=(20,20))
    plt.imshow(result)
    plt.savefig(os.path.join(out_path, 'coutour.png'))
    return contour

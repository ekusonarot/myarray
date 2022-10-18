import sys
import math
import numpy as np
from numpy.linalg import solve

def calc_contact_coord_ray_and_circle(theta, r):
    #光線と円が衝突する座標を計算
    theta = theta/180*math.pi
    if 30 < r:
        return 1000, 1000
    return r*math.cos(theta), r*math.sin(theta)

def calc_center(coord1, coord2):
    #二つの座標を通る円を求める．
    #2つの円が求められるが，原点から遠い円の中心座標を返す
    x1, y1, x2, y2 = coord1[0], coord1[1], coord2[0], coord2[1]
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    if dist == 0 or 2 < dist:
        return None
    a = (dist**2)/2*dist
    h = math.sqrt(1-a**2)
    x3 = x1+a*(x2-x1)/dist
    y3 = y1+a*(y2-y1)/dist
    x4 = x3+h*(y2-y1)/dist
    y4 = y3-h*(x2-x1)/dist

    x5=x3-h*(y2-y1)/dist
    y5=y3+h*(x2-x1)/dist
    if x4**2+y4**2 < x5**2+y5**2:
        return (x5, y5)
    else:
        return (x4, y4)


def main(lines):
    m = int(lines[0])
    lines = lines[1:]
    coords = [None] * m #光線と衝突した座標を格納
    center_coords = [None] * 10 #円の中心座標を格納
    cnt = 0
    for i, v in enumerate(lines):
        theta, r = float(v.split()[0]), float(v.split()[1])
        coords[i] = calc_contact_coord_ray_and_circle(theta, r)
    for i in range(-1, m):
        #二つの衝突座標を通る円の中心を求め
        #center_coordsに同じ円とみなされる座標が格納されていなければ
        #center_coordsに求めた円の座標を格納する
        #円の中心座標の距離が4以上であれば，別の円とみなす
        center = calc_center(coords[i-1], coords[i])
        if center == None:
            continue
        for i in range(10):
            if center_coords[i] == None:
                center_coords[i] = center
                cnt += 1
                break
            if math.sqrt((center_coords[i][0]-center[0])**2 + (center_coords[i][1]-center[1])**2) <= 4.:
                break
    
    print(cnt)
    for i in range(cnt):
        print("{} {}".format(center_coords[i][0], center_coords[i][1]))
    

if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)

import numpy as np
import cv2
import os
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from collections import deque
import itertools
import json

from homography.EM import EM
from homography.tag_points import Tag

def point_in_allpoints(extra_point, points, sides):
	point = Point(extra_point[0], extra_point[1])

	for side in sides:
		polygon = Polygon([(points[side[0]][0],points[side[0]][1]),(points[side[1]][0],points[side[1]][1]), 
							(points[side[2]][0],points[side[2]][1]),(points[side[3]][0],points[side[3]][1])])
		if polygon.contains(point):
			return True
	return False

def apply_vp(EM_class, points, sides, parallel_lines):
	parameter_points, parameter_lines = EM_class.parametrize_vanishing_points(points, sides, parallel_lines)
	points = EM_class.apply_vanishing_points(points, parallel_lines, parameter_points, parameter_lines)

	known_parameters = parameter_points.copy()

	vanishing_points = []
	for lines in parameter_lines:
		intersection = EM_class.line_intersection(points[ lines[0][0] ], points[ lines[0][1] ], points[ lines[1][0] ], points[ lines[1][1] ])
		vanishing_points.append(intersection)

	all_points = list(np.arange(len(points)))

	intersection_points = []
	for i in range(3):
		missing_points = []
		for line in parallel_lines[i]:
			missing_points.extend(line)
		intersection_points.append(list(set(all_points).difference(missing_points)))

	for i in range(3):
		int_point0 = intersection_points[i]
		int_point1 = intersection_points[i-1]
		if len(int_point0) == 0 or len(int_point1) == 0:
			continue

		keep_going = True
		counter = 0
		while keep_going:

			int_point1 = int_point1[::-1]

			for p0,p1 in zip(int_point0, int_point1):
				extra_point = EM_class.line_intersection(points[ p0 ], vanishing_points[i], points[ p1 ], vanishing_points[i-1])

				if point_in_allpoints(extra_point, points, sides):
					points = np.concatenate((points, extra_point.reshape(1,2)), axis = 0)
					index = len(points)-1
					parallel_lines[i].append([p0, index])
					parallel_lines[i-1].append([p1, index])
					keep_going = False

			if counter == 100:
				return None, None, None
			counter += 1

		if len(int_point1) == 2:
			parallel_lines[i-2].append([len(points)-1, len(points)-2])

	return points, sides, parallel_lines

def reorder_points(points, img_points, tags):
	ordered_points = []
	if points[-1][1] < points[-2][1]:
		top_back_point = points[-1]
		bottom_back_point = points[-2]
	else:
		top_back_point = points[-2]
		bottom_back_point = points[-1]
	if 'Front' in tags and 'Right' in tags:
		ordered_points.append(img_points['Front Left Top'])	
		ordered_points.append(img_points['Front Left Bottom'])	
		ordered_points.append(img_points['Front Right Top'])	
		ordered_points.append(img_points['Front Right Bottom'])	
		ordered_points.append(top_back_point)
		ordered_points.append(bottom_back_point)
		ordered_points.append(img_points['Back Right Top'])	
		ordered_points.append(img_points['Back Right Bottom'])
	if 'Front' in tags and 'Left' in tags:
		ordered_points.append(img_points['Back Left Top'])	
		ordered_points.append(img_points['Back Left Bottom'])	
		ordered_points.append(img_points['Front Left Top'])	
		ordered_points.append(img_points['Front Left Bottom'])
		ordered_points.append(top_back_point)
		ordered_points.append(bottom_back_point)
		ordered_points.append(img_points['Front Right Top'])	
		ordered_points.append(img_points['Front Right Bottom'])
	if 'Back' in tags and 'Right' in tags:
		ordered_points.append(img_points['Front Right Top'])	
		ordered_points.append(img_points['Front Right Bottom'])	
		ordered_points.append(img_points['Back Right Top'])	
		ordered_points.append(img_points['Back Right Bottom'])
		ordered_points.append(top_back_point)
		ordered_points.append(bottom_back_point)
		ordered_points.append(img_points['Back Left Top'])	
		ordered_points.append(img_points['Back Left Bottom'])
	if 'Back' in tags and 'Left' in tags:
		ordered_points.append(img_points['Back Right Top'])	
		ordered_points.append(img_points['Back Right Bottom'])	
		ordered_points.append(img_points['Back Left Top'])	
		ordered_points.append(img_points['Back Left Bottom'])	
		ordered_points.append(top_back_point)
		ordered_points.append(bottom_back_point)
		ordered_points.append(img_points['Front Left Top'])	
		ordered_points.append(img_points['Front Left Bottom'])		

	return np.asarray(ordered_points)
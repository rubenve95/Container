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

folder = 'MVI_4627.MP4'
data_path = 'data'

EM = EM(data_path, folder)

def get_zone_labels():
    return np.array([[238 ,201, 0],#yellow
                    [205, 92, 92],  #red 
                    [132 ,112 ,255],  #blue
                    [34 ,139 ,34], #green
                    [0,0,0]]) # black

def read_polygons(name):
	with open(name, 'rb') as f:
		array = np.load(f)
		array2 = np.load(f)
	return array, array2

def set_images(file_name):
	#print(file_name)
	EM.img = cv2.imread(os.path.join(data_path, 'images', folder, file_name))
	EM.img_map = cv2.imread(os.path.join(data_path, 'ground_truth', folder, file_name.split('.')[0] + '.png'))
	# cv2.imshow('image',EM.img)
	# cv2.imshow('image',EM.img_map)
	# key = cv2.waitKey(0)
	# if key == ord('q') & 0xFF:
	# 	cv2.destroyAllWindows()
	# 	exit()

def get_polygon_contour(img, color_index, show=False):

	color = get_zone_labels()[color_index]
	#print(color)
	for i in range(3):
		img[:,:,i][img[:,:,i] != color[i]] = 0

	if not img.any():
		return None

	img = cv2.resize(img, (600,400))

	if show:
		cv2.imshow('yo', img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#img = np.pad(img, (EM.padding,EM.padding), 'constant', constant_values=(0,0))

	thresh = img
	if show:
		cv2.imshow('threshold ',thresh)

	# dilate thresholded image - merges top/bottom 
	kernel = np.ones((3,3), np.uint8)
	dilated = cv2.dilate(thresh, kernel, iterations=3)
	if show:
		cv2.imshow('threshold dilated',dilated)

	# find contours
	contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	# simplify contours
	try:
		epsilon = 0.05*cv2.arcLength(contours[0],True)
	except:
		return None
	iters = 0
	while True:
		if iters == 100:
			return None
		iters += 1
		approx = cv2.approxPolyDP(contours[0],epsilon,True)
		if len(approx) > 4:
			epsilon *= (2**(1/iters))
		elif len(approx) < 4:
			epsilon /= (2**(1/iters))
		else:
			break
	if show:
		cv2.drawContours(img, [approx], 0, (255,255,255), 3)
		cv2.imshow('image',img)
		key = cv2.waitKey(0)
		if key == ord('q') & 0xFF:
			cv2.destroyAllWindows()
			exit()
	approx = approx.reshape(4,2)
	return approx

def get_polygons(img, show=False):
	polygons = []
	colors = []
	for color_index in range(4):
		polygon = get_polygon_contour(img.copy(), color_index, show=show)
		if polygon is not None:
			polygons.append(polygon)
			colors.append(color_index)

	return polygons, colors

def combine_polygons(polygons):
	if len(polygons) == 0:	
		print('No polygons found')
		return None
	elif len(polygons) == 1:
		return polygons
	elif len(polygons) == 2:
		matches = EM.nearest_neighbour(polygons[0], polygons[1])
		p = {}
		p['c0'] = {0: matches[0][0], 1: matches[1][0]}
		p['c1'] = {0: matches[0][1], 1: matches[1][1]}
		return EM.merge_polygon(p, polygons)
		
	elif len(polygons) == 3:
		matches = []
		matches.append(EM.nearest_neighbour(polygons[0], polygons[1]))
		matches.append(EM.nearest_neighbour(polygons[0], polygons[2]))
		matches.append(EM.nearest_neighbour(polygons[1], polygons[2]))

		p = {}
		p['c0'] = {0: matches[0][0][0], 1: matches[0][1][0]}
		p['c1'] = {0: matches[0][0][1], 1: matches[0][1][1]}

		keys = list(p.keys())
		for i in range(2):
			same_corner = False
			for key in keys:
				if matches[1][0][i] == p[key][0]:
					p[key][2] = matches[1][1][i]
					same_corner = True
			if not same_corner:
				if 'c2' not in p:
					p['c2'] = {0: matches[1][0][i], 2: matches[1][1][i]}
				else:
					p['c3'] = {0: matches[1][0][i], 2: matches[1][1][i]}

		keys = list(p.keys())
		for i in range(2):
			same_corner = False
			for key in keys:
				if 1 in p[key] and matches[2][0][i] == p[key][1]:
					p[key][2] = matches[2][1][i]
					same_corner = True
				elif 2 in p[key] and matches[2][1][i] == p[key][2]:
					p[key][2] = matches[2][0][i]
					same_corner = True
			if not same_corner:
				if 'c3' not in p:
					p['c3'] = {1: matches[2][0][i], 2: matches[2][1][i]}

		return EM.merge_polygon(p, polygons)

def point_in_allpoints(extra_point, points, sides):
	point = Point(extra_point[0], extra_point[1])

	for side in sides:
		polygon = Polygon([(points[side[0]][0],points[side[0]][1]),(points[side[1]][0],points[side[1]][1]), 
							(points[side[2]][0],points[side[2]][1]),(points[side[3]][0],points[side[3]][1])])
		if polygon.contains(point):
			return True
	return False

def apply_vp(points, sides, parallel_lines):
	parameter_points, parameter_lines = EM.parametrize_vanishing_points(points, sides, parallel_lines)
	points = EM.apply_vanishing_points(points, parallel_lines, parameter_points, parameter_lines)

	known_parameters = parameter_points.copy()

	vanishing_points = []
	for lines in parameter_lines:
		intersection = EM.line_intersection(points[ lines[0][0] ], points[ lines[0][1] ], points[ lines[1][0] ], points[ lines[1][1] ])
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
				extra_point = EM.line_intersection(points[ p0 ], vanishing_points[i], points[ p1 ], vanishing_points[i-1])

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

def polygon_on_edge(polygons):
	x_min, y_min = EM.width,EM.height
	x_max, y_max = 0,0

	for p in polygons:
		x_min = min(np.amin(p[:,0]), x_min)
		y_min = min(np.amin(p[:,1]), y_min)
		x_max = max(np.amax(p[:,0]), x_max)
		y_max = max(np.amax(p[:,1]), y_max)

	return (x_min == 0 or y_min == 0 or x_max == EM.width-1 or y_max == EM.height-1)

def save_vertices(points_dict):
	folder_path = os.path.join(data_path, 'results/ground_truth/vertices')
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	full_path = os.path.join(folder_path, folder + '.json')
	with open(full_path, 'w') as f:
		json.dump(points_dict, f)

def save_gt_polygons(filename, polygons, face_labels):
	folder_path = os.path.join(data_path, 'results/ground_truth/polygons', folder)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	with open(os.path.join(folder_path, filename + '.npy'), 'wb') as f:
		np.save(f, np.asarray(polygons))
		np.save(f, np.asarray(face_labels))

def save(points, sides, parallel_lines, filename):
	folder_path = os.path.join(data_path, 'results', 'box_ground_truth', folder)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	full_path = os.path.join(folder_path, filename + '.npy')
	sides = np.asarray(sides)
	parallel_lines = np.asarray(parallel_lines)

	with open(full_path, 'wb') as f:
		np.save(f, points)
		np.save(f, sides)
		np.save(f, parallel_lines)

def assess_points(old_points, new_points):
	nb_points = old_points.shape[0]
	#new_points_temp = new_points[:nb_points]
	distance = np.linalg.norm(old_points - new_points[:nb_points])
	#print(old_points, new_points[:nb_points])
	#print(distance)
	return distance

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

def convert_gt_folder():
	print('converting folder', folder)
	points_dict = {}
	files = os.listdir(os.path.join(data_path, 'images', folder))
	for file in tqdm(files):
		#print(file)
		# if file != '221.png':
		#  	continue
		set_images(file)
		polygons, labels = get_polygons(EM.img_map, show=False)
		if len(polygons) == 1:
			if not polygon_on_edge(polygons):
				save_gt_polygons(file, polygons, labels)
			print(file, 'only one side visible')
			continue

		new_polygons = combine_polygons(polygons)
		# best_polygons = EM.copy_polygons(polygons)

		if polygon_on_edge(new_polygons):
			print(file, 'is on edge')
			continue

		best_ordered_points = None
		best_distance = 1e6
		tagger = Tag()
		for i in range(len(polygons)):
			points, sides, parallel_lines = EM.define_box(new_polygons)
			#if i == 0:
				#EM.show_lines(points, sides, parallel_lines, show_indices=True, add_padding=False)

			try:
				new_points, sides, parallel_lines = apply_vp(points.copy(), sides, parallel_lines)
				#save(new_points, sides, parallel_lines, file)
			except:
				print('Vanishing point constraints failed', file)
				new_polygons.append(new_polygons.pop(0))
				labels.append(labels.pop(0))
				#EM.show_lines(points, sides, parallel_lines, show_indices=True, add_padding=False, filepath='221_vpfail.png')
				continue
			distance = assess_points(points, new_points)
			if distance < best_distance:
				#EM.show_lines(new_points, sides, parallel_lines, show_indices=True, add_padding=False)
				polygons_temp = EM.points_to_polygons(new_points, sides)
				img_points, tags = tagger(polygons_temp, labels)
				best_ordered_points = reorder_points(new_points, img_points, tags)
				best_distance = distance
				save_gt_polygons(file, new_polygons, labels)

			new_polygons.append(new_polygons.pop(0))
			labels.append(labels.pop(0))
		if best_ordered_points is not None:
			points_dict[file] = [{'vertices': best_ordered_points.tolist()}]
		#EM.show_lines(ordered_points, sides, parallel_lines, show_indices=True, add_padding=False)
		#print(img_points, face_labels)
	print('{} out of {} successful'.format(len(points_dict), len(files)))
	save_vertices(points_dict)

def make_cuboid_images():
	print('making images for:', folder)
	points_dict = {}
	files = os.listdir(os.path.join(data_path, 'images', folder))
	for file in tqdm(files):
		polygons, face_labels = read_polygons(os.path.join(data_path, 'results/polygons_vp', folder, file + '.npy'))
		polygons -= 100
		if len(polygons) > 1:
			set_images(file)
			points, sides, parallel_lines = EM.define_box(polygons)

			new_points, sides, parallel_lines = apply_vp(points.copy(), sides, parallel_lines)
			try:
				EM.show_lines(new_points, sides, parallel_lines, show_indices=None, add_padding=False, show_vp=False, show_gt=False, filepath=os.path.join('data/results/yoyoyo', folder, 'cuboids', file))
			except:
				print('vp failed for', file)

if __name__=="__main__":
	convert_gt_folder()
	#make_cuboid_images()

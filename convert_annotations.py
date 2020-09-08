import numpy as np
import cv2
import os

from homography.EM import EM

EM = EM('data', 'MVI_3018.MP4')

def read_polygons(name):
	with open(name, 'rb') as f:
		array = np.load(f)

	return array

def set_images(file_name, folder_name):
	#print(file_name)
	EM.img = cv2.imread(os.path.join('/home/ruben/Documents/Mavisoft/container_faces/data/images', folder_name, file_name))
	EM.img_map = cv2.imread(os.path.join('/home/ruben/Documents/Mavisoft/container_faces/data/ground_truth', folder_name, file_name))

def convert(polygons):

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

	# print(p)

	new_polygons = EM.merge_polygon(p, polygons)

	# print(new_polygons)

	return new_polygons

if __name__=="__main__":

	folder = 'MVI_3018.MP4'
	files = os.listdir('/home/ruben/Documents/Mavisoft/container_faces/data/results/polygons/' + folder)
	for file in files:
		name = '/home/ruben/Documents/Mavisoft/container_faces/data/results/polygons/MVI_3018.MP4/' + file
		set_images('.'.join(file.split('.')[:-1]), folder)
		polygons = read_polygons(name)
		if polygons.shape[0] < 3:
			continue
		new_polygons = convert(polygons)

		assert np.array_equal(polygons, new_polygons)

		points, sides, parallel_lines = EM.define_box(new_polygons)

		#print(points, sides, parallel_lines)
		EM.show_lines(points, sides, parallel_lines)
		break
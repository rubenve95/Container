import numpy as np
import cv2
import yaml
import os
import time

#4:black
#3:side green
#2:front
#top yellow and side red

class Tag():
	def __init__(self):

		self.sides = {2:'Front',
						1: 'Left',#Could be different
						3: 'Right',
						0: 'Top'}

		self.side_name_order = [('Front', 'Back'), ('Left', 'Right'), ('Top', 'Bottom')]

		self.adjacency = {'Front': 'Back', 'Back': 'Front', 'Left': 'Right', 
						'Right': 'Left', 'Top': 'Bottom', 'Bottom': 'Top'}

		self.tag_order = ['Front Left Bottom', 'Front Right Bottom', 'Front Right Top', 'Front Left Top',
							'Back Left Top', 'Back Right Top', 'Back Right Bottom', 'Back Left Bottom']

	def pretty_print(self,dict):
		print(yaml.dump(dict, default_flow_style=False))

	def show_points(self,point_dict, img_name, folder_name, write=False):
		def draw_circle_and_text(img, text, coordinates, color):
			if color == 'red':
				rgb = (0,0,255)
			else:
				rgb = (0,255,0)

			img = cv2.circle(img, coordinates, 10, rgb, -1)

			text_coordinates = (coordinates[0] - 50, coordinates[1] - 20)
			cv2.putText(img,text, 
					text_coordinates, 
					cv2.FONT_HERSHEY_SIMPLEX, 
					0.5,
					rgb,
					1)

		path = os.path.join('data/images/',folder_name, img_name)
		# path = os.path.join('data/results/segmentation_post/MVI_3018.MP4', img_name)
		img = cv2.imread(path)
		img = cv2.resize(img, (600,400))
		border_margin = 60
		img = cv2.copyMakeBorder(img, border_margin, border_margin, border_margin, border_margin, cv2.BORDER_CONSTANT)

		for label in point_dict:
			#for point in point_dict[label]:
			point = point_dict[label]
			coordinates = (point[0],point[1])
			draw_circle_and_text(img, label, coordinates, 'green')
		self.show_image(img, img_name, write=write, folder_name=folder_name)

	def show_image(self, img, name = 'image', write=True, folder_name=None):
		if write:
			path = os.path.join('data/results/homographies', folder_name, name + '_tags.png')
			cv2.imwrite(path,img)
		else:
			cv2.imshow(name, img)
			key = cv2.waitKey(0)
			if key == ord('q') & 0xFF:
				cv2.destroyAllWindows()
				exit()
			cv2.destroyAllWindows()
	# def path_order(self, side):
	# 	if side == 'Front' or side == 'Back':
	# 		base_order = [side + ' Left Top', side + ' Right Top', side + ' Right Bottom', side + ' Left Bottom']
	# 	if side == 'Left' or side == 'Right':
	# 		base_order = ['Front ' + side + ' Top', 'Front ' + side + ' Bottom', 'Back ' + side + ' Bottom', 'Back ' + side + ' Top']
	# 	if side == 'Top':
	# 		base_order = ['Front Left ' + side, 'Front Right ' + side, 'Back Right ' + side, 'Back Left ' + side]
	# 	return base_order

	def ints_to_labels(self, sides):
		sides = [self.sides[x] for x in sides]
		return sides

	def find_connected_sides(self, point, ipoly, polygons, sides):
		connected_sides = []
		for ipoly2,poly in enumerate(polygons):
			if ipoly == ipoly2:
				continue
			for ip,p in enumerate(poly):
				if np.array_equal(point, p):
					connected_sides.append(sides[ipoly2])
		return connected_sides

	def swap_tags(self, points):
		'''
		Only works for swapping Front to Back
		'''
		replace = ['Front', 'Back', 'Left', 'Right']
		points_copy = points.copy()
		points = {}
		for tag in points_copy:
			sides = tag.split(' ')
			new_sides = []
			for s in sides:
				if s in replace:
					new_sides.append(self.adjacency[s])
				else:
					new_sides.append(s)
			new_tag = " ".join(new_sides)
			points[new_tag] = points_copy[tag]
		return points

	def reorder_corner_name(self, corner_name):

		def sort_side(elem):
			for i,s in enumerate(self.side_name_order):
				if elem in s:
					return i

		side_list = corner_name.split(' ')
		side_list = sorted(side_list, key = sort_side)
		corner_name = " ".join(side_list)
		return corner_name

	def unconnected_corner(self, main_side, sides):
		corner_side = [self.adjacency[s] if s != main_side else s for s in sides]
		corner_name = " ".join(corner_side)
		corner_name = self.reorder_corner_name(corner_name)
		return corner_name

	def connected_corner(self, main_side, connected_sides, sides):
		connected_sides.append(main_side)
		not_connected_side = list(set(sides).difference(connected_sides))
		if len(not_connected_side) > 0:
			connected_sides.append(self.adjacency[not_connected_side[0]])
		corner_name = " ".join(connected_sides)
		corner_name = self.reorder_corner_name(corner_name)
		return corner_name

	def missing_side_name(self,corner_name):
		side_names = corner_name.split(' ')
		for side_tuple in self.side_name_order:
			if not side_tuple[0] in side_names and not side_tuple[1] in side_names:
				return side_tuple
		return None

	def handle_doubles(self, img_points, sides):
		labels = list(img_points.keys())
		for label in labels:
			if len(img_points[label]) > 1:
				p0 = img_points[label][0]
				p1 = img_points[label][1]
				missing_side_tuple = self.missing_side_name(label)
				if missing_side_tuple == ('Top', 'Bottom'):
					label0,label1 = missing_side_tuple if p0[1] < p1[1] else (missing_side_tuple[1],missing_side_tuple[0])	
				elif missing_side_tuple == ('Left', 'Right'):				
					label0,label1 = missing_side_tuple if p0[0] < p1[0] else (missing_side_tuple[1],missing_side_tuple[0])
				else:
					other_side_name = 'Left' if 'Left' in sides else 'Right'
					if (other_side_name == 'Left' and p0[0] > p1[0]) or (other_side_name == 'Right' and p0[0] < p1[0]):
						label0,label1 = missing_side_tuple
					else:
						label0,label1 = (missing_side_tuple[1],missing_side_tuple[0])

				label0,label1 = label + ' ' + label0, label + ' ' + label1
				label0,label1 = self.reorder_corner_name(label0), self.reorder_corner_name(label1)				
				img_points[label0] = p0
				img_points[label1] = p1
				img_points.pop(label, None)
			else:
				img_points[label] = img_points[label][0]

	def handle_front_or_back(self, sides, polygons):
		try:
			index = sides.index('Front')
			if 'Left' in sides:
				other_side = 'Left'
				other_index = sides.index(other_side)
			else:
				other_side = 'Right'
				other_index = sides.index(other_side)#Will throw exception if not existing
			average_front = [int(p[0]) for p in polygons[index]]
			average_front = sum(average_front)/len(average_front)
			average_other = [int(p[0]) for p in polygons[other_index]]
			average_other = sum(average_other)/len(average_other)
			if (other_side == 'Right' and average_other < average_front) or (other_side == 'Left' and average_other > average_front):
				sides[index] = 'Back'
			return sides
		except:
			return sides

	def one_side(self, side, polygon):
		polygon = sorted(polygon, key = lambda k: k[1])
		bottom_points = polygon[-2:]
		top_points = polygon[:-2]
		bottom_points = sorted(bottom_points, key = lambda k: k[0])
		top_points = sorted(top_points, key= lambda k: k[0])

		img_points = {}
		if side == 'Left':
			img_points['Back Left Bottom'] = bottom_points[0]
			img_points['Front Left Bottom'] = bottom_points[1]
			img_points['Back Left Top'] = top_points[0]
			img_points['Front Left Top'] = top_points[1]
		elif side == 'Right':
			img_points['Front Right Bottom'] = bottom_points[0]
			img_points['Back Right Bottom'] = bottom_points[1]
			img_points['Front Right Top'] = top_points[0]
			img_points['Back Right Top'] = top_points[1]
		elif side == 'Front':
			img_points['Front Left Bottom'] = bottom_points[0]
			img_points['Front Right Bottom'] = bottom_points[1]
			img_points['Front Left Top'] = top_points[0]
			img_points['Front Right Top'] = top_points[1]
		elif side == 'Back':
			img_points['Back Right Bottom'] = bottom_points[0]
			img_points['Back Left Bottom'] = bottom_points[1]
			img_points['Back Right Top'] = top_points[0]
			img_points['Back Left Top'] = top_points[1]
		elif side == 'Top':	#This is impossible to know
			raise NotImplementedError
		return img_points

	def compute_ratios(self,points,sides):

		def sort_by_order(elem):
			return self.tag_order.index(elem)

		scores = []
		for side in sides:
			side_tags = [tag for tag in points if side in tag.split(' ')]
			side_tags = sorted(side_tags, key = sort_by_order)

			point0 = np.asarray(points[side_tags[0]])
			point1 = np.asarray(points[side_tags[1]])
			point2 = np.asarray(points[side_tags[2]])
			point3 = np.asarray(points[side_tags[3]])
			ratio_0 = np.linalg.norm(point0 - point1)/np.linalg.norm(point2 - point3)
			ratio_1 = np.linalg.norm(point0 - point3)/np.linalg.norm(point1 - point2)
			ratio_score = (abs(1 - ratio_0) + abs(1 - ratio_1))/2
			scores.append(ratio_score)
		return scores

	def __call__(self, polygons, sides):
		sides = self.ints_to_labels(sides)
		sides = self.handle_front_or_back(sides, polygons)

		if len(sides) == 1:
			img_points = self.one_side(sides[0], polygons[0])
		else:
			img_points = {}

			for ipoly,polygon in enumerate(polygons):
				for point in polygon:
					connected_sides = self.find_connected_sides(point, ipoly, polygons, sides)
					if len(connected_sides) == 0:
						corner_name = self.unconnected_corner(sides[ipoly], sides)
					else:
						corner_name = self.connected_corner(sides[ipoly], connected_sides, sides)
					if corner_name not in img_points:
						img_points[corner_name] = []
					new_point = (int(point[0]),int(point[1]))
					if len(img_points[corner_name]) == 0 or new_point != img_points[corner_name][0]:
						img_points[corner_name].append(new_point)
				
			# self.pretty_print(img_points)
			self.handle_doubles(img_points, sides)

		# self.pretty_print(self.img_points)
		#self.show_points(img_points, img_name)

		return img_points, sides
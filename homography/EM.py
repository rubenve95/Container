import numpy as np
import os
from matplotlib.path import Path
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
import time
from project_code.saver.saver import ArraySaver

#4:black
#3:side green
#2:front
#top yellow and side red

class EM():
	def __init__(self, data_root, folder_name, height=400, width = 600, padding=100, num_classes=5):
		self.folder_name = folder_name
		self.data_root = data_root

		self.width = width
		self.height = height
		self.padding = padding
		self.num_classes = num_classes

	def softmax(self, x, axis=None):
		x = x - x.max(axis=axis, keepdims=True)
		y = np.exp(x)
		return y / y.sum(axis=axis, keepdims=True)

	def get_numpy_array(self, name):
		path = os.path.join(self.data_root, 'results/segmentation_probs', self.folder_name, name)
		with open(path, 'rb') as f:
			array = np.load(f)[0]
		return array

	def define_box(self, polygons):
		'''
		Assuming polygons length >= 2
		direction computed from last 2 points
		'''
		points = []
		sides = []
		parallel_lines = []

		for poly in polygons:
			sides.append([])
			for point in poly:

				point_index = len(points)
				for j,p in enumerate(points):
					if np.array_equal(point, p):
						point_index = j
						break

				sides[-1].append(point_index)
				if point_index == len(points):
					points.append(point)

		for side in sides:
			new_groups = [ [[side[0],side[1]], [side[2],side[3]]], [[side[1],side[2]], [side[3],side[0]]] ] 
			for new_group in new_groups:
				isnew = True

				for group in parallel_lines:
					if new_group[0] in group or new_group[0][::-1] in group:
						group.append(new_group[1])
						isnew = False
						break
					elif new_group[1] in group or new_group[1][::-1] in group:
						group.append(new_group[0])
						isnew = False
						break
				if isnew:
					parallel_lines.append(new_group)

		points = np.asarray(points)
		return points, sides, parallel_lines

	def parametrize_vanishing_points(self, points, sides, parallel_lines):
		parameter_points = sides[0].copy()
		#direction_line = None
		parameter_lines = [[],[],[]]
		for i,group in enumerate(parallel_lines):
			for line in group:
				first_point_in = (line[0] in parameter_points)
				second_point_in = (line[1] in parameter_points)

				if first_point_in and second_point_in:
					parameter_lines[i].append(line)

				elif (first_point_in and not second_point_in) or (not first_point_in and second_point_in):
					if len(parameter_lines[i]) == 0:
						parameter_lines[i].append(line)
						new_point = line[1] if first_point_in else line[0]
						parameter_points.append(new_point)
					else:
						parameter_lines[i].append(line)
						break
		return parameter_points, parameter_lines

	def apply_vanishing_points(self, points, parallel_lines, parameter_points, parameter_lines):
		known_parameters = parameter_points.copy()
		new_points = points.copy()

		vanishing_points = []
		for lines in parameter_lines:
			intersection = self.line_intersection(points[ lines[0][0] ], points[ lines[0][1] ], points[ lines[1][0] ], points[ lines[1][1] ])
			vanishing_points.append(intersection)

		while len(known_parameters) < len(points):
			lines_almost_known = [[],[],[]]
			findable_point = []
			for i,lines in enumerate(parallel_lines):
				for line in lines:
					first_point_in = (line[0] in known_parameters)
					second_point_in = (line[1] in known_parameters)
					if (first_point_in and not second_point_in) or (not first_point_in and second_point_in):
						lines_almost_known[i].append(line)
						new_point = line[1] if first_point_in else line[0]
						findable_point.append(new_point)
			findable_point = list(set([x for x in findable_point if findable_point.count(x) > 1]))[0]
			intersection_lines = []
			for i,lines in enumerate(lines_almost_known):
				for line in lines:
					for j,coor in enumerate(line):
						if findable_point == coor:
							known_point = new_points[ line[j-1] ]
							intersection_lines.append([known_point, vanishing_points[i]])
			intersection = self.line_intersection(intersection_lines[0][0], intersection_lines[0][1], intersection_lines[1][0], intersection_lines[1][1])
			new_points[findable_point] = intersection
			known_parameters.append(findable_point)

		return new_points

	def get_direction_angle(self, points, parameter_lines):
		line = parameter_lines[-1][-1]
		vector = points[line[0]] - points[line[1]]
		angle = np.arctan(vector)
		return angle

	def apply_direction_angle(self, points, parameter_lines, angle):
		angle = np.radians(angle)
		line = parameter_lines[-1][-1]
		vector = points[line[1]] - points[line[0]]
		r = np.array(( (np.cos(angle), -np.sin(angle)),
			(np.sin(angle),  np.cos(angle)) ))
		points[line[1]] = points[line[0]] + r.dot(vector)
		return points

	def EM_box(self, points, sides, parallel_lines, parameter_points, parameter_lines, faces, show=False, total_iters=6, convergence_ratio=1e-3):
		#Denk later nog aan richting, zit op het moment niet in parameter_points

		points = self.apply_vanishing_points(points, parallel_lines, parameter_points, parameter_lines)
		best_points = points.copy()
		parameter_order = np.asarray(parameter_points.copy())
		xy_order = np.arange(2)
		order = np.array(np.meshgrid(parameter_order, xy_order)).T.reshape(-1,2)
		order = np.vstack((order, np.array([-1, -1])))

		image_offset = int(0.05*self.width)
		best_p_list = []
		best_p_av = 1e9
		step_size = max(int((self.width + self.height + 4*self.padding)/20), 1)
		t_prev = time.time()
		for it in range(total_iters):
			t_now = time.time()
			t_prev = t_now
			np.random.shuffle(order)
			prev_best_p_av = best_p_av
			best_p_list = []
			step_size = max(int(step_size/4), 1)
			time0 = time.time()

			for o in order:
				points = best_points.copy()
				best_p = 1e9
				param = o[0]
				xy = o[1]

				if o[0] < 0:
					mini = -5
					maxi = 4.1
					used_step_size = 0.1/(it+1)
				else:
					mini = int(max(0, points[param][xy] - image_offset))
					maximum_img = self.width+2*self.padding-1 if xy == 0 else self.height+2*self.padding-1
					maxi = int(min(maximum_img, points[param][xy] + image_offset))
					used_step_size = step_size
				#before = points[param][xy]
				for value_var in np.arange(mini,maxi+1,used_step_size):
					new_points = points.copy()
					p_list = []
					p=0
					if o[0] < 0:
						new_points = self.apply_direction_angle(new_points, parameter_lines, value_var)
						#self.show_lines(new_points, sides, parallel_lines, special_point = param, show_indices=False, name='angle')
					else:
						new_points[param][xy] = value_var
					new_points = self.apply_vanishing_points(new_points, parallel_lines, parameter_points, parameter_lines)
					for side,face in zip(sides,faces):
						polygon = np.asarray([new_points[s] for s in side])
						p_list.append(self.probability(face, polygon))#, show= (show and (value_var%20 == 0))))

					p = sum(p_list)
					#if value_var == before:
						# print('before', p)
						# print(new_points)
						# print(points)

						# p += self.probability(face, polygon, show= (show and (value_var%20 == 0)))
					# if show and (value_var % 20 == 0 or o[0] < 0):
					# 	self.show_lines(new_points, sides, parallel_lines, special_point = param, show_indices=False, name='during')
					if p <= best_p:
						best_p = p
						best_points = new_points.copy()
				best_p_list.append(best_p)

			best_p_av = sum(best_p_list)/len(best_p_list)
			#print("Iteration {} Best likelihood {:.2f} Time {:.2f}".format(it, best_p_av, time.time() - time0))
			if -best_p_av + prev_best_p_av < best_p_av*convergence_ratio and step_size == 1:
				#print(it)
				#self.show_lines(new_points, sides, parallel_lines, special_point = param, show_indices=False, name='during')
				return best_points
		return best_points

	def opt_vp(self, polygons, face_labels, array, show=False, filepath=None):

		if len(polygons) == 1:
			return polygons

		# self.height,self.width = 400,600
		# self.padding = 60
		array = self.softmax(array, axis=0)
		faces = []
		for f in face_labels:
			new_face = self.add_padding_face(array[f])
			faces.append(new_face)

		points, sides, parallel_lines = self.define_box(polygons)
		orig_points = points.copy()
		parameter_points, parameter_lines = self.parametrize_vanishing_points(points, sides, parallel_lines)

		best_points = self.EM_box(points, sides, parallel_lines, parameter_points, parameter_lines, faces, show=show)

		if show:
			#while True:
			self.show_lines(orig_points, sides, parallel_lines, show_indices=False, name='before', filepath=filepath)
			self.show_lines(best_points, sides, parallel_lines, show_indices=False, name='after', filepath=filepath)

		best_polygons = self.points_to_polygons(best_points, sides)

		return best_polygons

		# new_points = self.apply_vanishing_points(points, parallel_lines, parameter_points, parameter_lines)

		#self.show_lines(points, sides, parallel_lines)

	def points_to_polygons(self, points, sides):
		polygons = []
		for side in sides:
			polygon = np.asarray([points[s] for s in side])
			polygons.append(polygon)
		return polygons

	def show_lines(self, points, sides, parallel_lines, special_point = None, show_indices = False, name='img', filepath=None):
		img = self.img.copy()
		img_map = self.img_map.copy()
		img = cv2.resize(img, (self.width, self.height))
		img = cv2.copyMakeBorder(img, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT)
		img_map = cv2.resize(img_map, (self.width, self.height))
		img_map = cv2.copyMakeBorder(img_map, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT)

		colors = [(255,0,0),(0,255,0),(0,0,255)]

		# for i,p in enumerate(points):
		# 	text = str(i) if show_indices else str(p[0]) + ' ' + str(p[1])
		# 	color = (255,255,255) if special_point == i else (0,255,0)
		# 	cv2.putText(img,text, (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX,1,color,1)
		# 	cv2.putText(img_map,text, (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX,1,color,1)

		for ilines,lines in enumerate(parallel_lines):
			if len(lines) == 1:
				continue
			for i,line in enumerate(lines):

				line0 = [ points[ line[0] ], points[ line[1] ] ]
				line1 = [ points[ lines[i-1][0] ], points[ lines[i-1][1] ] ]

				intersection = self.line_intersection(line0[0],line0[1], line1[0],line1[1])

				try:
					intersection = (int(intersection[0]), int(intersection[1]))
					img = cv2.line(img, (int(line0[0][0]),int(line0[0][1])), intersection, colors[ilines], 1) 
					img = cv2.line(img, (int(line0[1][0]),int(line0[1][1])), (int(line0[0][0]),int(line0[0][1])), colors[ilines], 3) 

					img = cv2.line(img, (int(line1[0][0]),int(line1[0][1])), intersection, colors[ilines], 1) 
					img = cv2.line(img, (int(line1[1][0]),int(line1[1][1])), (int(line1[0][0]),int(line1[0][1])), colors[ilines], 3)

					img_map = cv2.line(img_map, (int(line0[0][0]),int(line0[0][1])), intersection, colors[ilines], 1) 
					img_map = cv2.line(img_map, (int(line0[1][0]),int(line0[1][1])), (int(line0[0][0]),int(line0[0][1])), colors[ilines], 3) 

					img_map = cv2.line(img_map, (int(line1[0][0]),int(line1[0][1])), intersection, colors[ilines], 1) 
					img_map = cv2.line(img_map, (int(line1[1][0]),int(line1[1][1])), (int(line1[0][0]),int(line1[0][1])), colors[ilines], 3)  
				except:
					return
					#print(intersection)

		img = np.hstack((img, img_map))
		if filepath is None:
			cv2.imshow(name, img)
			key = cv2.waitKey(0)
			if key == ord('q') & 0xFF:
				cv2.destroyAllWindows()
				exit()
		else:
			path_name = filepath.split('/')
			path_name[-1] = path_name[-1].split('.')[0] + '_' + name + '.png'
			filepath = '/'.join(path_name)
			cv2.imwrite(os.path.join('data/results/yoyoyo', filepath), img)

	def line_intersection(self, a1,a2, b1,b2):

		def perp( a ) :
			b = np.empty_like(a)
			b[0] = -a[1]
			b[1] = a[0]
			return b

		da = a2-a1
		db = b2-b1
		dp = a1-b1
		dap = perp(da)
		denom = np.dot( dap, db)
		num = np.dot( dap, dp )
		return (num / denom.astype(float))*db + b1

	def show_image(self, mask, polygon, name='img'):
		#img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		#print(polygon)
		img = self.img_map.copy()
		img = cv2.resize(img, (self.width, self.height))
		img = cv2.copyMakeBorder(img, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT)
		# img = np.asarray([np.pad(img[...,i], (self.padding,self.padding), 'constant', constant_values=(0,0)) for i in range(img.shape[2])])
		# img = np.transpose(img, (1,2,0))
		#img = np.pad(img, (self.padding,self.padding,0), 'constant', constant_values=(0,0,0))
		#img[...,1] = np.pad(img[...,1], (self.padding,self.padding), 'constant', constant_values=(0,0))
		#img[...,2] = np.pad(img[...,2], (self.padding,self.padding), 'constant', constant_values=(0,0))
		#img[...,2] = mask*255
		#img[...,1] = mask*255
		# img[...,0] = mask*255
		img[...,2][mask] = 255
		img = cv2.circle(img, (polygon[0][0], polygon[0][1]), 3, (0,255,0), -1)
		cv2.putText(img,str(polygon[0][0]) + ' ' + str(polygon[0][1]), (polygon[0][0], polygon[0][1]), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
		img = cv2.circle(img, (polygon[1][0], polygon[1][1]), 3, (0,255,0), -1)
		cv2.putText(img,str(polygon[1][0]) + ' ' + str(polygon[1][1]), (polygon[1][0], polygon[1][1]), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
		img = cv2.circle(img, (polygon[2][0], polygon[2][1]), 3, (0,255,0), -1)
		cv2.putText(img,str(polygon[2][0]) + ' ' + str(polygon[2][1]), (polygon[2][0], polygon[2][1]), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
		img = cv2.circle(img, (polygon[3][0], polygon[3][1]), 3, (0,255,0), -1)
		cv2.putText(img,str(polygon[3][0]) + ' ' + str(polygon[3][1]), (polygon[3][0], polygon[3][1]), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
		if img.shape != self.img.shape:
			self.img = cv2.resize(self.img, (self.width, self.height))
			self.img = cv2.copyMakeBorder(self.img, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT)
			# self.img = np.asarray([np.pad(self.img[...,i], (self.padding,self.padding), 'constant', constant_values=(0,0)) for i in range(img.shape[2])])
			# self.img = np.transpose(self.img, (1,2,0))
			# self.img[...,1] = np.pad(self.img[...,1], (self.padding,self.padding), 'constant', constant_values=(0,0))
			# self.img[...,2] = np.pad(self.img[...,2], (self.padding,self.padding), 'constant', constant_values=(0,0))
		img = np.hstack((img, self.img))		
		cv2.imshow(name, img)
		key = cv2.waitKey(0)
		if key == ord('q') & 0xFF:
			cv2.destroyAllWindows()
			exit()

	def get_faces_probability(self, array):
		#Return acummulated probabilities for face labels and for background
		means = np.mean(array, axis=(1,2))
		return means[:-1], means[-1]

	def get_most_probable_face(self, array, index):
		p_faces, p_black = self.get_faces_probability(array)
		faces_order = np.argsort(p_faces)[::-1]
		face_index = faces_order[index]
		if p_faces[face_index]/p_black < 0.02:
			return None
		return face_index

	def get_most_probable_point(self,face):
		blurred_face = cv2.blur(face,(7,7))
		ind = np.unravel_index(np.argmax(blurred_face.T, axis=None), (self.width,self.height))
		return ind

	def get_generic_polygon(self):
		x_min = int(0.25*self.width - 1)
		x_max = int(0.75*self.width - 1)
		y_min = int(0.25*self.height - 1)
		y_max = int(0.75*self.height - 1)

		polygon = np.asarray([(x_min,y_min),
								(x_min,y_max),
								(x_max,y_max),
								(x_max,y_min)])
		return polygon

	def get_probable_polygon(self, face):
		ind = self.get_most_probable_point(face)
		x = ind[0]
		y = ind[1]
		offset = 0.05
		x_min = int(max(0, x - offset*self.width))
		x_max = int(min(self.width-1, x + offset*self.width))
		y_min = int(max(0, y - offset*self.height))
		y_max = int(min(self.height-1, y + offset*self.height))

		polygon = np.asarray([(x_min,y_min),
					(x_min,y_max),
					(x_max,y_max),
					(x_max,y_min)])
		return polygon

	def get_random_polygon(self):
		while True:
			random_x = np.random.randint(0, self.width-1, size=2)
			random_y = np.random.randint(0, self.height-1, size=2)
			if random_x[0] != random_x[1] and random_y[0] != random_y[1]:
				break
		x_min = min(random_x[0], random_x[1])
		x_max = max(random_x[0], random_x[1])
		y_min = min(random_y[0], random_y[1])
		y_max = max(random_y[0], random_y[1])

		polygon = np.asarray([(x_min,y_min),
			(x_min,y_max),
			(x_max,y_max),
			(x_max,y_min)])
		return polygon

	def get_polygon_contour(self, face, show=False):
		# threshold image
		img = face.copy()
		img = (img*255).astype(np.uint8)
		ret,thresh = cv2.threshold(img,127,255,0)
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

	def get_polygon_gradient(self,face, show=False):
		horizontal = np.sum(face, axis=0)
		grad_hor = np.gradient(horizontal, 5)

		vertical = np.sum(face, axis=1)
		grad_ver = np.gradient(vertical, 5)

		if show:
			f = plt.figure()
			plt.plot(grad_ver, label='vertical')
			plt.plot(grad_hor, label='horizontal')
			plt.legend()
			plt.show()

		#As the gradient goes up when the label shows up, argmax for first value, argmin for second value
		x_min = np.argmax(grad_hor)
		x_max = np.argmin(grad_hor)
		y_min = np.argmax(grad_ver)
		y_max = np.argmin(grad_ver)

		polygon = np.asarray([(x_min,y_min),
									(x_min,y_max),
									(x_max,y_max),
									(x_max,y_min)])
		return polygon

	def get_initial_polygon(self,face, show=False):
		polygon = self.get_polygon_contour(face, show=show)

		if polygon is not None and not self.small_area(polygon, threshold=0.05):
			return polygon

		polygon = self.get_polygon_gradient(face, show=show)

		if not self.polygon_self_crosses([polygon]) or self.polygon_convex([polygon]) and not self.small_area(polygon, threshold=0.05):
			return polygon

		polygon = self.get_probable_polygon(face)
		return polygon

	def polygon_mask(self, polygon, show=False):
		x, y = np.meshgrid(np.arange(self.width + 2*self.padding), np.arange(self.height + 2*self.padding))
		x, y = x.flatten(), y.flatten()
		points = np.vstack((x,y)).T

		p = Path(polygon)
		grid = p.contains_points(points)
		mask = grid.reshape(self.height + 2*self.padding,self.width + 2*self.padding)
		if show:
			#self.show_image((mask*255).astype(np.uint8), polygon)
			self.show_image(mask, polygon)
		return mask

	def probability(self, face, polygon, show=False, mask=None):
		if mask is None:
			mask = self.polygon_mask(polygon, show=False)

		if show:
			self.show_image(mask, polygon)
			# self.show_image(mask*face, polygon)
		#P_in = np.sum(face[mask==1])
		#P_out = np.sum(1 - face[mask==0])

		P_in = -np.sum(np.log(face[mask==1]))
		P_out = -np.sum(np.log(1 - face[mask==0]))

		#print(P_in, P_out)

		# P_in = -np.sum(np.log(face[mask==1]))
		# P_out = -np.sum(np.log(1 - face[mask==0]))
		P = P_in + P_out
		# area = np.sum(mask)/np.prod(mask.shape)
		# if area < 0.1:
		# 	P *= area
		return P#/np.sum(mask[mask==1])

	def point_distance(self, p0, p1):
		return np.linalg.norm(p0-p1)

	def point_too_close(self, polygon, index):
		for i in range(polygon.shape[0]-1):
			if self.point_distance(polygon[index-i-1], polygon[index]) < 4:
				return True
		return False

	def value_on_line(self, pi, p0, p1, var):
		not_var = 1 - var
		numerator = p1[not_var] - p0[not_var]
		if numerator == 0:
			#Parallel, every value fine in this case
			return 0
		l_factor = (pi[not_var] - p0[not_var])/numerator
		ultimate_value = p0[var] + l_factor*(p1[var] - p0[var])
		return ultimate_value

	def max_min_value(self, polygon, index, var):
		#Accept only values for x or y for which the quadrilateral polygon stays convex
		#That is, when no angles of >=180 are formed
		pi = polygon[index]
		p0 = polygon[index-1]
		p1 = polygon[index-2]
		p2 = polygon[index-3]

		values = []
		values.append(self.value_on_line(pi,p0,p1,var))
		values.append(self.value_on_line(pi,p0,p2,var))
		values.append(self.value_on_line(pi,p1,p2,var))

		smaller = [v for v in values if v <= pi[var]]
		greater = [v for v in values if v >= pi[var]]

		max_image = self.height-1 if var==1 else self.width-1
		max_image += 2*self.padding

		maxi = min(min(greater), max_image) if greater != [] else max_image
		mini = max(max(smaller), 0) if smaller != [] else 0

		if maxi != max_image:
			maxi = math.floor(maxi-0.001)
		else:
			maxi = int(maxi)
		if mini != 0:
			mini = math.ceil(mini + 0.001)
		else:
			mini = int(mini)

		return mini,maxi

	def small_area(self, polygon, threshold=0.01):
		mask = self.polygon_mask(polygon)
		area = np.sum(mask)/(self.width*self.height)
		if area < threshold:
			return True
		return False

	def handle_shared_points(self, order, polygons):
		shared_points = {}
		skip_list = []
		for io,o in enumerate(order):
			if io in skip_list:
				continue
			poly_nb = o[0]
			point_nb = o[1]
			if poly_nb not in shared_points:
				shared_points[poly_nb] = {}
			shared_points[poly_nb][point_nb] = {}

			for ipoly,p in enumerate(polygons):
				for ipt,pt in enumerate(p):
					if np.array_equal(polygons[poly_nb][point_nb], pt):
						shared_points[poly_nb][point_nb][ipoly] = ipt

						if ipt != point_nb and ipoly != poly_nb:
							try:
								index = order.tolist().index([ipoly,ipt,0])
								skip_list.append(index)
								index = order.tolist().index([ipoly,ipt,1])
								skip_list.append(index)
							except:
								continue
		skip_list = list(set(skip_list))
		skip_list.sort()
		for i in skip_list[::-1]:
			order = np.delete(order, i, 0)
		return shared_points, order

	def process_face(self, faces, polygons, nb_retries=0, total_iters=6, convergence_ratio=1e-3, show=False):
		'''
		Create the most probable polygon for a face
		By iteratively choosing one coordinate of the polygon which gives the best probability
		In goes:
			- face: Probability array for a certain face label
			- polygon: An Nx2 array of x,y coordinates
			- nb_retries: Indicates if the algorithm is run again after failure. 
						Should be set to 0 upon first try
			- total_iters: How often the main loop can maximally be run
			- convergence_ratio: Stop if the improvement in probability upon last iteration is smaller than this ratio
		Returns:
			- polygon: An Nx2 array of x,y coordinates
		'''
		point_order = np.arange(polygons[0].shape[0])
		xy_order = np.arange(2)
		polygon_order = np.arange(len(polygons))
		order = np.array(np.meshgrid(polygon_order, point_order, xy_order)).T.reshape(-1,3)
		shared_points, order = self.handle_shared_points(order, polygons)
		best_p_av = 1e9
		step_size_factor = 0.1#/(len(polygons))
		max_steps = int(20/len(polygons))
		for it in range(total_iters):
			step_size_factor /= 4
			prev_best_p_av = best_p_av
			best_p_list = []
			np.random.shuffle(order)
			for o in order:
				best_p = 1e9
				poly_nb = o[0]
				point_nb = o[1]
				var = o[2]
				best_value_var = polygons[poly_nb][point_nb][var]

				mini, maxi = 0, max(self.width+2*self.padding,self.height+2*self.padding)
				for po_nb in shared_points[poly_nb][point_nb]:
					pt_nb = shared_points[poly_nb][point_nb][po_nb]
					new_mini,new_maxi = self.max_min_value(polygons[po_nb], pt_nb, var)
					mini = max(mini,new_mini)
					maxi = min(maxi,new_maxi)
				t0 = time.time()

				step_size = max(int((self.width + self.height + 4*self.padding)/2*step_size_factor), 1)
				mini = max(polygons[poly_nb][point_nb][var] - step_size*max_steps, mini)
				maxi = min(polygons[poly_nb][point_nb][var] + step_size*max_steps, maxi)

				for value_var in range(mini,maxi+1,step_size):
					p = 0
					for po_nb in shared_points[poly_nb][point_nb]:
						pt_nb = shared_points[poly_nb][point_nb][po_nb]
						polygons[po_nb][pt_nb][var] = value_var
						p += self.probability(faces[po_nb], polygons[po_nb], show= (show and (value_var%20 == 0)))
					if p <= best_p:
					# if p >= best_p:
						best_p = p
						best_value_var = value_var

				t2 = time.time()
				#print("loop time", t2 - t0)

				for po_nb in shared_points[poly_nb][point_nb]:
					pt_nb = shared_points[poly_nb][point_nb][po_nb]
					polygons[po_nb][pt_nb][var] = best_value_var
				best_p_list.append(best_p)

			best_p_av = sum(best_p_list)/len(best_p_list)
			# print("Iteration {} Best likelihood {}".format(it, best_p_av))
			if -best_p_av + prev_best_p_av < best_p_av*convergence_ratio:
				if len(polygons) == 1 and (self.small_area(polygons[0])):
					#print("Failure!")
					if nb_retries == 0:
						polygons[0] = self.get_generic_polygon()
						polygons[0] += self.padding
						polygons = self.process_face(faces, polygons, nb_retries=1)
					if nb_retries == 1:
						polygons[0] = self.get_random_polygon()
						polygons[0] += self.padding
						polygons = self.process_face(faces, polygons, nb_retries=2)
					else:
						polygons[0] = None
				return polygons
		return polygons

	def set_images(self,img_name):
		img_path = os.path.join(self.data_root, 'images', self.folder_name, img_name)
		self.img = cv2.imread(img_path)
		#self.img = cv2.resize(self.img, (600, 400))

		img_map_path = os.path.join(self.data_root, 'results/segmentation_raw', self.folder_name, img_name)
		self.img_map = cv2.imread(img_map_path)
		#self.img_map = self.img_map[:,:self.width]

	def nearest_neighbour(self, p0, p1):
		'''
		Nearest Neighbours matching two out of four points between two polygons
		In goes:
			- p0,p1: two polygons with four points each
		Returns:
			- best_comb: Tuple with two arrays of two indices matched with nearest neighbours
				First index of the first tuple is of the first polygon.
				It matches to the first index of the second tuple, which belongs to the second polygon.
		'''
		nbs = np.arange(p0.shape[0])
		combs0 = np.array(np.meshgrid(nbs, nbs)).T.reshape(-1,2)
		combs1 = np.asarray([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])

		best_distance = self.width*1000
		best_comb = None
		for c0 in combs0:
			if c0[0] == c0[1]:
				continue
			for c1 in combs1:
				p0_comb = p0[c0]
				p1_comb = p1[c1]
				distance = np.linalg.norm(p0_comb[0]-p1_comb[0]) + np.linalg.norm(p0_comb[1]-p1_comb[1])
				if distance < best_distance:
					best_distance = distance
					best_comb = (c0, c1)
		return best_comb

	def merge_polygon(self, p, polygons):
		new_polygons = self.copy_polygons(polygons)

		for corner in p:
			x = 0
			y = 0
			for point in p[corner]:
				poly_point = new_polygons[point][p[corner][point]]
				x += poly_point[0]
				y += poly_point[1]
			x = int(x/len(p[corner]))
			y = int(y/len(p[corner]))

			for point in p[corner]:
				new_polygons[point] [p[corner][point]] [0] = x
				new_polygons[point] [p[corner][point]] [1] = y
		return new_polygons

	def copy_polygons(self, polygons):
		new_polygons = []
		for p in polygons:
			new_polygons.append(np.copy(p))
		return new_polygons

	def calculate_overlap(self, mask1, mask2):
		overlap = mask2.copy()
		overlap = overlap[mask1 == True]
		return overlap

	def remove_full_overlap(self, polygons):
		masks = [self.polygon_mask(p) for p in polygons]
		remove = []
		for im1,m1 in enumerate(masks):
			for im2,m2 in enumerate(masks):
				if im1 == im2:
					continue
				overlap = self.calculate_overlap(m1,m2)
				if np.sum(overlap)/np.prod(overlap.shape) == 1.0:
					remove.append(im1)
		return remove

	def mask_overlap(self, masks):
		for im,m1 in enumerate(masks[:-1]):
			for m2 in masks[im+1:]:
				overlap = self.calculate_overlap(m1, m2)
				if np.sum(overlap)/np.prod(overlap.shape) > 0.01:
					return True
		return False

	def polygon_self_crosses(self, polygons):
		for poly in polygons:
			bowtie = Polygon(poly)
			if not bowtie.is_valid:
				return True
		return False

	def polygon_convex(self, polygons):
		r=0.001
		for poly in polygons:
			for ip,pt in enumerate(poly):
				poly_3point = np.delete(poly, ip, 0)
				p = Path(poly_3point)
				if p.contains_point(pt,radius=r) or p.contains_point(pt,radius=-r):
					return True
		return False

	def remove_polygons(self, polygons, face_labels, remove):
		remove.sort()
		for r in remove[::-1]:
			polygons.pop(r)
			face_labels.pop(r)
		return polygons, face_labels

	def merge_corners(self, polygons, faces):
		#Merges corners of polygons so that polygons connect and form one object

		#list for each polygon for each index to be filled with list of coordinate tuples
		if len(polygons) == 1:
			return polygons

		if len(polygons) == 2:

			nbs = np.arange(4)
			combs = np.array(np.meshgrid(nbs, nbs)).T.reshape(-1,2)

			p = {}
			best_probability = 1e9
			best_polygons = self.copy_polygons(polygons)

			for i0,c0 in enumerate(combs[:-1]):
				p['c0'] = {0: c0[0], 1: c0[1]}

				for c1 in combs[i0+1:]:

					p['c1'] = {0: c1[0], 1: c1[1]}
					if p['c1'][0]%2 == p['c0'][0]%2 or p['c1'][1]%2 == p['c0'][1]%2: continue

					new_polygons = self.merge_polygon(p, polygons)

					if self.polygon_self_crosses(new_polygons) or self.polygon_convex(new_polygons):
						continue

					probability = 0
					masks = []
					for face,poly in zip(faces,new_polygons):
						mask = self.polygon_mask(poly)
						masks.append(mask)
						probability += self.probability(face, poly, mask=mask)

					if (probability <= best_probability and not self.mask_overlap(masks)):
						best_probability = probability
						best_polygons = self.copy_polygons(new_polygons)
			return best_polygons

		if len(polygons) == 3:

			nbs = np.arange(4)
			combs_corner3 = np.array(np.meshgrid(nbs, nbs, nbs)).T.reshape(-1,3)
			combs_corner2 = np.array(np.meshgrid(nbs, nbs)).T.reshape(-1,2)

			p = {}

			best_probability = 1e9
			best_polygons = self.copy_polygons(polygons)

			for c0 in combs_corner3:
				p['c0'] = {0: c0[0], 1: c0[1], 2: c0[2]}

				for c1 in combs_corner2:
					p['c1'] = {0: c1[0], 1: c1[1]}
					if p['c1'][0]%2 == p['c0'][0]%2 or p['c1'][1]%2 == p['c0'][1]%2: continue

					for c2 in combs_corner2:
						p['c2'] = {0: c2[0], 2: c2[1]}
						if p['c2'][0]%2 == p['c0'][0]%2 or p['c2'][2]%2 == p['c0'][2]%2: continue
						if p['c2'][0] == p['c1'][0] or p['c2'][0]%2 != p['c1'][0]%2: continue

						for c3 in combs_corner2:
							p['c3'] = {1: c3[0], 2: c3[1]}
							if p['c3'][1]%2 == p['c0'][1]%2 or p['c3'][2]%2 == p['c0'][2]%2: continue
							if p['c3'][1] == p['c1'][1] or p['c3'][2] == p['c2'][2]: continue
							if p['c3'][1]%2 != p['c1'][1]%2 or p['c3'][2]%2 != p['c2'][2]%2: continue

							new_polygons = self.merge_polygon(p, polygons)

							if self.polygon_self_crosses(new_polygons) or self.polygon_convex(new_polygons):
								continue

							probability = 0
							masks = []
							for face,poly in zip(faces,new_polygons):
								mask = self.polygon_mask(poly)
								masks.append(mask)
								probability += self.probability(face, poly, mask=mask)
							if (probability <= best_probability and not self.mask_overlap(masks)):
								best_probability = probability
								best_polygons = self.copy_polygons(new_polygons)
			return best_polygons

	def add_padding_face(self, face):

		#was 0.49, now 0.499999

		pad_value = 0.49	#0.5 means it doesn't matter being in padding or not. 0.49 to slightly punish it
		padded_face = np.pad(face, (self.padding,self.padding), 'constant', constant_values=(pad_value,pad_value))
		return padded_face

	def get_segmentation_output(self, polygons, face_labels):
		output = (np.ones((self.height,self.width))*(self.num_classes-1)).astype(np.uint8)

		for i,poly in enumerate(polygons):
			mask = self.polygon_mask(poly, show=False)
			mask = mask[self.padding:-self.padding,self.padding:-self.padding]
			output[mask] = face_labels[i]
		return output

	def EM(self, array, use_softmax=True):
		if use_softmax:
			array = self.softmax(array, axis=0)

		self.num_classes,self.height,self.width = array.shape
		#self.padding = int(0.1*max(self.height,self.width))

		polygons = []
		face_labels = []
		faces = []
		#output = (np.ones(array[0].shape)*(self.num_classes-1)).astype(np.uint8)#background label
		for i in range(3):
			face_index = self.get_most_probable_face(array, i)
			if face_index is None:
				break
			face = array[face_index]
			polygon = self.get_initial_polygon(face, show=False)
			face = self.add_padding_face(face)
			polygon += self.padding
			polygon = self.process_face([face], [polygon], show=False)[-1]
			if polygon is not None:
				polygons.append(polygon)
				face_labels.append(face_index)
				faces.append(face)

		if len(polygons) > 1:

			t0 = time.time()

			remove = self.remove_full_overlap(polygons)
			polygons,face_labels = self.remove_polygons(polygons, face_labels, remove)

			# print('before merge')
			# for p in polygons:
			# 	mask = self.polygon_mask(p, show=True)
			polygons = self.merge_corners(polygons, faces)

			# print('after merge')
			# for p in polygons:
			# 	mask = self.polygon_mask(p, show=True)
			t1 = time.time()
			polygons = self.process_face(faces, polygons, show=False)

			# print('after opt')
			# for p in polygons:
			# 	mask = self.polygon_mask(p, show=True)

		remove = []
		for i in range(len(polygons)):
			if self.small_area(polygons[i]):
				remove.append(i)
		polygons,face_labels = self.remove_polygons(polygons, face_labels, remove)

			#print("Merging time", t1 - t0)
			# for p in polygons:
				# mask = self.polygon_mask(p, show=True)


		return polygons, face_labels

		#print("Combined processing time", time.time() - t1)
import numpy as np
import os
import cv2

class Homography():
	def __init__(self):
		self.width = 1920
		self.height = 1080

		self.container_longside = 20
		self.container_shortside = 8
		self.container_height = 8.6

		img_scale = 1#0.75
		self.scale = img_scale*self.width/self.container_longside
		self.container_longside *= self.scale
		self.container_shortside *= self.scale
		self.container_height *= self.scale

		#Good: Left<->Right Top<->Bottom
		self.container_points = {'Front Left Top': {'x': 0, 'y': 0, 'z': self.container_shortside},
						'Front Left Bottom': {'x': 0, 'y': self.container_height, 'z': self.container_shortside},
						'Front Right Top': {'x': self.container_longside, 'y': 0, 'z': self.container_shortside}, 
						'Front Right Bottom': {'x': self.container_longside, 'y': self.container_height, 'z': self.container_shortside}, 
						'Back Left Top': {'x': 0, 'y': 0, 'z':  0}, 
						'Back Left Bottom': {'x': 0, 'y': self.container_height, 'z': 0}, 
						'Back Right Top': {'x': self.container_longside, 'y': 0, 'z': 0}, 
						'Back Right Bottom': {'x': self.container_longside, 'y': self.container_height, 'z': 0}}

		#print(self.container_points)


		# self.container_points = {'Front Left Top': {'x': 0, 'y': 0, 'z': self.height/2},
		# 				'Front Left Bottom': {'x': 0, 'y': self.height/2, 'z': self.height/2},
		# 				'Front Right Top': {'x': self.width/2, 'y': 0, 'z': self.height/2}, 
		# 				'Front Right Bottom': {'x': self.width/2, 'y': self.height/2, 'z': self.height/2}, 
		# 				'Back Left Top': {'x': 0, 'y': 0, 'z':  0}, 
		# 				'Back Left Bottom': {'x': 0, 'y': self.height/2, 'z': 0}, 
		# 				'Back Right Top': {'x': self.width/2, 'y': 0, 'z': 0}, 
		# 				'Back Right Bottom': {'x': self.width/2, 'y': self.height/2, 'z': 0}}

		self.width_scaler = 1#self.width/600
		self.height_scaler = 1#self.height/400

	def calculate_homography(self, points, side):
		camera_pts, object_pts = self.create_point_matrices(points, side)
		if camera_pts.shape[0] == 4:
			M, mask = cv2.findHomography(camera_pts, object_pts)
			#M = cv2.getPerspectiveTransform(camera_pts, object_pts)
		else:
			M = None
		return M

	def create_point_matrices(self,points,side_label):
		camera_pts = []
		object_pts = []
		#labels = ['Front Left Top', 'Front Right Top', 'Front Right Bottom', 'Front Left Bottom']
		for label in points:
			corner_sides = label.split(' ')
			if side_label in corner_sides:
				x = self.width_scaler*(points[label][0])# - 60)
				y = self.height_scaler*(points[label][1])# - 60)
				camera_pts.append((x,y))
				if side_label == 'Front' or side_label == 'Back':
					obj_x = self.container_points[label]['x']# + 0.5*600#*self.container_longside/self.width
					obj_y = self.container_points[label]['y']# + 0.5*400#*self.container_height/self.height
				elif side_label == 'Left' or side_label == 'Right':
					obj_x = self.container_points[label]['z']# + 500
					obj_y = self.container_points[label]['y']# + 300
				else:
					obj_x = self.container_points[label]['x']# + 100
					obj_y = self.container_points[label]['z']# + 100
				object_pts.append((obj_x,obj_y))
		# print(object_pts)
		#print(camera_pts)
		return np.asarray(camera_pts).reshape(-1,2).astype(np.float32), np.asarray(object_pts).reshape(-1,2).astype(np.float32)

	def show_homography(self, M, img_name, vid_name, side, write = False):
		orig_img = cv2.imread(os.path.join('data/images/', vid_name, img_name))
		warp_img = cv2.resize(orig_img, (600,400))
		padding = 60
		warp_img = cv2.copyMakeBorder(warp_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
		img = cv2.warpPerspective(warp_img, M, (self.width, self.height))
		#M_inv = np.linalg.inv(M)
		#img = cv2.warpPerspective(img, M_inv, (self.width, self.height))
		#img = np.hstack((orig_img, img))
		#img = cv2.resize(img,(int(self.width/1.1), int(self.height/(1.1*2))))
		if write:
			folder = os.path.join('data/results/homographies', vid_name)
			if not os.path.exists(folder):
				os.mkdir(folder)
			cv2.imwrite(os.path.join(folder, img_name + '_' + side + '.png'), img)
		else:
			cv2.imshow(side + img_name, img)
			key = cv2.waitKey(0)
			if key == ord('q') & 0xFF:
				cv2.destroyAllWindows()
				exit()
			cv2.destroyAllWindows()

	# def homography_side(self, img_name, side):
	# 	M = None
	# 	camera_pts, object_pts = self.side_points(self.dict[img_name], side)
	# 	if camera_pts.shape[0] == 4:
	# 		M, mask = cv2.findHomography(camera_pts, object_pts, cv2.RANSAC,5.0)
	# 	return M

	# def apply_homographies(self, write = False):
	# 	for img_name in tqdm(self.dict):
	# 		sides = list(ADJACENCY.keys())
	# 		for side in sides:
	# 			if side != 'Left':
	# 				continue
	# 			M = self.homography_side(img_name, side)
	# 			if M is not None:
	# 				self.show_homography(M, img_name, side, write=write)

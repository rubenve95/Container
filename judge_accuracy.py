import numpy as np
import os
import cv2
import json
import itertools
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from homography.EM import EM
from homography.tag_points import Tag
from homography.utils import apply_vp, reorder_points
from project_code.dataloader.utils import decode_seg_map_sequence, encode_segmap
from project_code.metrics.metrics import Metrics
from project_code.saver.saver import ImgSaver

def show_img(img, name='img'):
	cv2.imshow(name, img)
	key = cv2.waitKey(0)
	if key == ord('q') & 0xFF:
		cv2.destroyAllWindows()
		exit()

def convert_polygons(points):
	polygons = []
	new_polygon = np.zeros((4,2))
	new_polygon[0] = points[0]
	new_polygon[1] = points[1]
	new_polygon[2] = points[3]
	new_polygon[3] = points[2]
	polygons.append(new_polygon)
	if points[6,0] > points[2,0]:
		new_polygon = np.zeros((4,2))
		new_polygon[0] = points[2]
		new_polygon[1] = points[3]
		new_polygon[2] = points[7]
		new_polygon[3] = points[6]
		polygons.append(new_polygon)

	top_point = Point(points[4])
	if not Polygon(polygons[-1]).contains(top_point) and not Polygon(polygons[-2]).contains(top_point):
		new_polygon = np.zeros((4,2))
		new_polygon[0] = points[0]
		new_polygon[1] = points[2]
		new_polygon[2] = points[6]
		new_polygon[3] = points[4]
		polygons.append(new_polygon)
	return polygons

def get_pck(points, folder, filename, alpha=0.1):
	#assumes 8 points in the right order

	dict_path = os.path.join('data/results/ground_truth/vertices', folder + '.json')

	with open(dict_path) as f:
		vertex_dict = json.load(f)
	vertices = vertex_dict[filename][0]['vertices']

	min_x = 1e6
	max_x = 0
	min_y = 1e6
	max_y = 0
	for point in vertices:
		min_x = min(min_x, point[0])
		max_x = max(max_x, point[0])
		min_y = min(min_y, point[1])
		max_y = max(max_y, point[1])

	width = max_x - min_x
	height = max_y - min_y
	max_dim = max(width, height)

	pck = []
	for vert,point in zip(vertices, points):
		pck.append(np.linalg.norm(vert - point) < alpha*max_dim)

	return float(sum(pck)/len(pck))

def calculate_accuracy(folder, filename):
	evaluator = Metrics(5)

	array_path = os.path.join('data/results_original/cuboids/polygons', folder, filename + '.npy')
	#array_path = os.path.join(array_path, filename + '.npy')
	gt_path = os.path.join('data/ground_truth', folder, filename)

	with open(array_path, 'rb') as f:
		points = np.load(f).T

		#face_labels = np.load(f)
	gt_img = cv2.imread(gt_path)[:,:,::-1]
	height, width, _ = gt_img.shape
	scaling_x, scaling_y = float(600/width), float(400/height)
	points[:,0] *= scaling_x
	points[:,1] *= scaling_y
	polygons = convert_polygons(points)
	#print(polygons)

	gt_img = cv2.resize(gt_img, (600,400), interpolation=cv2.INTER_NEAREST)
	gt_img = encode_segmap(gt_img, oneHot=False)

	post = EM(None, None, padding=0)

	face_labels = list(np.unique(gt_img))
	if len(face_labels) > len(polygons):
		face_labels.remove(4)
	face_label_permutations = list(itertools.permutations(face_labels))

	best_acc = 0
	best_miou = 0
	for face_labels in face_label_permutations:

		pred = post.get_segmentation_output(polygons, face_labels)
		pred = np.expand_dims(pred, axis=0)
		evaluator.reset()
		evaluator.add_batch(np.expand_dims(gt_img, axis=0), pred)
		accuracy = evaluator.Pixel_Accuracy()
		miou = evaluator.Mean_Intersection_over_Union()

		if accuracy > best_acc:
			best_acc = accuracy
			best_miou = miou
			pred = decode_seg_map_sequence(pred)

			#img_saver = ImgSaver()
			#img_saver(pred[0], filename, folder=os.path.join('cuboids/segmentations', folder))

	baseline_path = os.path.join('data/results_original/polygons_vp', folder, filename + '.npy')
	with open(baseline_path, 'rb') as f:
		baseline_polygons  = np.load(f)
		baseline_face_labels  = np.load(f)

	post = EM(None, None)

	baseline_pred = post.get_segmentation_output(baseline_polygons, baseline_face_labels)
	baseline_pred = np.expand_dims(baseline_pred, axis=0)
	evaluator.reset()
	evaluator.add_batch(np.expand_dims(gt_img, axis=0), baseline_pred)
	baseline_accuracy = evaluator.Pixel_Accuracy()
	baseline_miou = evaluator.Mean_Intersection_over_Union()

	if len(baseline_polygons) == 1:
		baseline_pck = 0
	else:
		baseline_polygons -= 100
		points, sides, parallel_lines = post.define_box(baseline_polygons)
		points, sides, parallel_lines = apply_vp(post, points, sides, parallel_lines)
		tagger = Tag()
		img_points, tags = tagger(baseline_polygons, baseline_face_labels)

		points = reorder_points(points, img_points, tags)
		baseline_pck = get_pck(points, folder, filename)

	return best_acc, baseline_accuracy, best_miou, baseline_miou, baseline_pck

def judge_cuboids():
	foldername = 'MVI_3015.MP4'
	path = os.path.join('data/results_original/cuboids/cuboid_images', foldername)
	files = os.listdir(path)
	acc = []
	miou = []
	baseline_acc = []
	baseline_miou = []
	baseline_pck = []
	for file in tqdm(files):
		print(file)
		new_acc, new_baseline_acc, new_miou, new_baseline_miou, new_baseline_pck = calculate_accuracy(foldername, file)
		acc.append(new_acc)
		baseline_acc.append(new_baseline_acc)
		miou.append(new_miou)
		baseline_miou.append(new_baseline_miou)
		baseline_pck.append(new_baseline_pck)
		print(file, new_acc, new_baseline_acc, new_miou, new_baseline_miou, new_baseline_pck)
	print('Accuracy Deep cuboid detector:', float(sum(acc)/len(acc)))
	print('Accuracy baseline:', float(sum(baseline_acc)/len(baseline_acc)))
	print('MIOU Deep cuboid detector:', float(sum(miou)/len(miou)))
	print('MIOU baseline:', float(sum(baseline_miou)/len(baseline_miou)))
	print('PCK baseline:', float(sum(baseline_pck)/len(baseline_pck)))

if __name__=="__main__":
	judge_cuboids()
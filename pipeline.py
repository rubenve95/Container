from tqdm import tqdm
import os
import numpy as np
import time
import cv2
import json
import statistics

from homography.EM import EM
from homography.homography import Homography
from homography.tag_points import Tag
from project_code.saver.saver import ImgSaver
from project_code.saver.saver import ArraySaver
from project_code.dataloader.utils import decode_seg_map_sequence, encode_segmap
from project_code.metrics.metrics import Metrics
from homography.utils import apply_vp

def DL_segmentation_acc(folder_name, current_step, results_folder):
	files = os.listdir(os.path.join('data/images', folder_name))
	files.sort()

	opt_class = EM(folder_name)
	evaluator = Metrics(5)
	pck_list = []
	results_dict = {}

	pbar = tqdm(files)
	for file_name in pbar:

		results_dict[file_name] = {}

		gt_path = os.path.join('data/ground_truth', folder_name, file_name)
		gt_img = cv2.imread(gt_path)[:,:,::-1]
		gt_img = cv2.resize(gt_img, (opt_class.width,opt_class.height), interpolation=cv2.INTER_NEAREST)
		gt_img = encode_segmap(gt_img, oneHot=False)

		pred_path = os.path.join('data/results/segmentation_raw', folder_name, file_name)
		pred = cv2.imread(pred_path)[:,:,::-1]
		pred = cv2.resize(pred, (opt_class.width,opt_class.height), interpolation=cv2.INTER_NEAREST)
		pred = encode_segmap(pred, oneHot=False)
		#pred = opt_class.get_segmentation_output(polygons, face_labels)
		pred = np.expand_dims(pred, axis=0)
		evaluator.add_batch(np.expand_dims(gt_img, axis=0), pred)

		results_dict[file_name]['accuracy'] = evaluator.Pixel_Accuracy()
		results_dict[file_name]['miou'] = evaluator.Mean_Intersection_over_Union()
		results_dict[file_name]['pck'] = pck_list
		evaluator.reset()
		pck_list = []

	with open(os.path.join('data', results_folder, 'stats', folder_name + '_' + current_step + '.json'), 'w+') as f:
		json.dump(results_dict, f)


def pck_for_files(folder_name, current_step, results_folder, save=True, padding=100):
	files = os.listdir(os.path.join('data/images', folder_name))
	files.sort()
	methods = ['contour', 'gradient', 'generic', 'random', 'quadrilaterals']

	if current_step in methods:
		return

	if current_step in methods:
		opt = EM(folder_name, init_method=current_step, padding=padding)
	else:
		opt = EM(folder_name, padding=padding)
	#evaluator = Metrics(5)
	#pck_list = []
	with open(os.path.join('data', results_folder, 'stats', folder_name + '_' + current_step + '.json'), 'r') as f:
		results_dict = json.load(f)

	print(folder_name, current_step)
	pbar = tqdm(files)
	for file_name in pbar:


		polygons,face_labels = load_polygons(os.path.join('data', results_folder, 'polygons', current_step, folder_name, file_name + '.npy'))

		#if polygons is None:
		#print(file_name, polygons, face_labels)

		pck_list = calculate_pck(polygons, face_labels, file_name, folder_name, results_folder, padding=padding)
		#print(pck_list)

		results_dict[file_name]['pck'] = pck_list.copy()

	with open(os.path.join('data', results_folder, 'stats', folder_name + '_' + current_step + '.json'), 'w') as f:
		json.dump(results_dict, f)
	#print(results_dict)

#Also add methods of post_process here when this part is finished
#Maybe add extra tags to polygons_foldername, indicating padding and stuff

def calculate_pck(polygons, face_labels, file_name, folder_name, results_folder, padding=0, alpha=0.1):
	#This pck is different than that of Deep Cuboid Detection, as it only tests visible points
	#Also, it tests on more images

	#Note also that when averaging pck over all folders, it is not a true pck anymore, as other folders have different amount of keypoints. 
	#Could call it average pck instead


	#path = os.path.join('data', results_folder, 'ground_truth', 'polygons', folder_name, file_name + '.npy')
	# if os.path.exists(path):
	# 	print('yay')
	# else:
	# 	print(path, 'does not exist')
	# try:	
	# 	#gt_polygons, gt_face_labels = load_polygons(path)#No padding right?
	# except:
	# 	return []
		#print('i')

	def mirror_tag(tag):
		label = tag.split(' ')
		label[0] = 'Front' if label[0] == 'Back' else 'Back'
		label[1] = 'Right' if label[1] == 'Left' else 'Left'
		return " ".join(label)

	with open(os.path.join('data', results_folder, 'ground_truth', 'points', folder_name + '.json'), 'r') as f:
		gt_dict = json.load(f)

	gt_img_points = gt_dict[file_name]

	tagger = Tag()
	#gt_img_points, _ = tagger(gt_polygons, gt_face_labels)
	try:
		img_points, _ = tagger(polygons, face_labels)
	except:
		#print([False]*len(gt_img_points))
		return [False]*len(gt_img_points)

	# print(img_points.keys())
	# print(gt_img_points.keys())


	min_x = 1e6
	max_x = 0
	min_y = 1e6
	max_y = 0
	for gt_tag in gt_img_points:
		for point in gt_img_points[gt_tag]:
			#point = gt_img_points[gt_tag]
			min_x = min(min_x, point['x'])
			max_x = max(max_x, point['x'])
			min_y = min(min_y, point['y'])
			max_y = max(max_y, point['y'])
	width = max_x - min_x
	height = max_y - min_y
	max_dim = max(width, height)

	pck = []
	for gt_tag in gt_img_points:
		if gt_img_points[gt_tag][0]['attribute'] != 'Corner':
			continue

		if gt_tag in img_points:
			point = np.asarray(img_points[gt_tag])
			point -= padding
			gt_point = np.asarray([gt_img_points[gt_tag][0]['x'],gt_img_points[gt_tag][0]['y']])
			if np.linalg.norm(point - gt_point) < alpha*max_dim:
				pck.append(True)
				continue
		pck.append(False)

	pck_mirror = []
	for gt_tag in gt_img_points:
		if gt_img_points[gt_tag][0]['attribute'] != 'Corner':
			continue

		mirror_gt_tag = mirror_tag(gt_tag)
		if mirror_gt_tag in img_points:
			point = np.asarray(img_points[mirror_gt_tag])
			point -= padding
			gt_point = np.asarray([gt_img_points[gt_tag][0]['x'],gt_img_points[gt_tag][0]['y']])
			if np.linalg.norm(point - gt_point) < alpha*max_dim:
				pck_mirror.append(True)
				continue
		pck_mirror.append(False)	
	return pck if sum(pck) >= sum(pck_mirror) else pck_mirror

def save_polygons_and_image(polygons, face_labels, prediction_map, folder_name, file_name, prediction_type, opt_class, results_folder):
	seg_dir = os.path.join('segmentations', prediction_type)
	dir_path = os.path.join('data', results_folder, seg_dir)
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
	img_saver = ImgSaver(results_folder=results_folder)
	pred = decode_seg_map_sequence(prediction_map)
	img_saver(pred[0], file_name, folder=os.path.join(seg_dir, folder_name))

	# if prediction_type in ['merged', 'combined_regular', 'combined_vp']:
	# 	img_dir = os.path.join('images', prediction_type)
	# 	dir_path = os.path.join('data/results', img_dir)
	# 	if not os.path.exists(dir_path):
	# 		os.mkdir(dir_path)
	# 	folder_path = os.path.join(img_dir, folder_name)
	# 	if not os.path.exists(folder_path):
	# 		os.mkdir(folder_path)
	# 	opt_class.set_images(file_name, set_gt=False)
	# 	points, sides, parallel_lines = opt_class.define_box(polygons)
	# 	opt_class.show_lines(points, sides, parallel_lines, special_point = None, show_indices = False, 
	# 		filepath=os.path.join(folder_path, file_name), add_padding=True, show_gt=False, show_vp=False)

	poly_dir = os.path.join('polygons', prediction_type)
	dir_path = os.path.join('data', results_folder, poly_dir)
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
	array_saver = ArraySaver(results_folder=results_folder)
	array_saver(polygons, file_name + '.npy', folder=os.path.join(poly_dir, folder_name), array1=face_labels)

def add_to_evaluator(opt_class, evaluator, polygons, face_labels, folder_name, file_name):

	gt_path = os.path.join('data/ground_truth', folder_name, file_name)
	gt_img = cv2.imread(gt_path)[:,:,::-1]
	gt_img = cv2.resize(gt_img, (opt_class.width,opt_class.height), interpolation=cv2.INTER_NEAREST)
	gt_img = encode_segmap(gt_img, oneHot=False)

	pred = opt_class.get_segmentation_output(polygons, face_labels)
	pred = np.expand_dims(pred, axis=0)
	evaluator.add_batch(np.expand_dims(gt_img, axis=0), pred)

	return pred

# def load_polygons(file_name, folder_name, polygons_name, results_folder):
# 	path = os.path.join('data', results_folder, 'polygons', polygons_name, folder_name, file_name + '.npy')
# 	with open(path, 'rb') as f:
# 		polygons = np.load(f)
# 		face_labels = np.load(f)
# 	return polygons, face_labels

def load_polygons(path):
	with open(path, 'rb') as f:
		polygons = np.load(f)
		face_labels = np.load(f)
	return polygons, face_labels

def show_image(folder_name, current_step, results_folder, padding=100, save=True):
	files = os.listdir(os.path.join('data/images', folder_name))
	files.sort()
	opt_class = EM(folder_name, padding=padding)
	#print(opt_class.padding)

	if save:
		img_dir = os.path.join('images', current_step)
		dir_path = os.path.join('data', results_folder, img_dir)
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)
		folder_path = os.path.join(dir_path, folder_name)
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)

	pbar = tqdm(files)
	print(folder_name, results_folder, current_step)
	for file_name in pbar:
		pbar.set_description("Processing %s" % file_name)
		filepath = os.path.join(folder_path, file_name)
		#if os.path.exists(filepath):
			#continue
		# else:
			# print(filepath)
		polygons, face_labels = load_polygons(os.path.join('data', results_folder, 'polygons', current_step, folder_name, file_name + '.npy'))
		polygons -= padding
		opt_class.set_images(file_name, set_gt=False)
		if current_step in ['DL_segmentation']:
			break

		elif len(polygons) == 1 or current_step.split('_')[0] in ['quadrilaterals','contour','gradient','generic','random']:
			if current_step.split('_')[-1] == 'before':
				polygons += padding
			opt_class.show_polygons(polygons, name=file_name, filepath=filepath)

		else:
			#print(polygons, face_labels)
			points, sides, parallel_lines = opt_class.define_box(polygons)
			#print(points,sides,parallel_lines)

			if current_step in ['cuboid_vp', 'cuboid_post']:
				try:
					new_points, new_sides, new_parallel_lines = apply_vp(opt_class, points, sides, parallel_lines)
					# print('this then')
					if new_points is not None:
						points,sides,parallel_lines = new_points, new_sides, new_parallel_lines
						# print('okay')
				except:
					print('VP failed')
					#print(points,sides,parallel_lines)

			try:
				opt_class.show_lines(points, sides, parallel_lines, show_indices=None, add_padding=False, show_vp=False, show_gt=False, name=file_name, filepath=filepath)
			except:
				try:
					opt_class.show_polygons(polygons, name=file_name, filepath=filepath)
				except:
					print('Could not make image for:', filepath)
			#print('did it work?')

def make_complete_image(folder_name, results_folder, methods):
	files = os.listdir(os.path.join('data/images', folder_name))
	files.sort()
	#opt_class = EM(folder_name, padding=0)
	pbar = tqdm(files)
	print(folder_name, results_folder)

	for file_name in pbar:
		pbar.set_description("Processing %s" % file_name)

		top_images = []
		bottom_images = []

		#opt_class.set_images(file_name, set_gt=False)
		#image_orig = cv2.resize(opt_class.img, (opt_class.width, opt_class.height))
		#_images.append(image_orig)

		#gt_path = os.path.join('data/ground_truth', folder_name, file_name)
		#gt_img = cv2.imread(gt_path)[:,:,::-1]
		#gt_img = cv2.resize(gt_img, (opt_class.width,opt_class.height), interpolation=cv2.INTER_NEAREST)
		#_images.append(gt_img)

		if results_folder != 'results_init':
			top_images.append(cv2.imread(os.path.join('data/results_original/segmentation_raw', folder_name, file_name)))

		for method in methods:
			bottom_images.append(cv2.imread(os.path.join('data', results_folder, 'images', method, folder_name, file_name)))
			if results_folder == 'results_init':
				top_images.append(cv2.imread(os.path.join('data', results_folder, 'images', method + '_before', folder_name, file_name)))

		write_path = os.path.join('data', results_folder, 'images', 'all')
		if not os.path.exists(write_path):
			os.mkdir(write_path)
		write_path = os.path.join(write_path, folder_name)
		if not os.path.exists(write_path):
			os.mkdir(write_path)
		write_path = os.path.join(write_path, file_name)

		try:
			images = np.concatenate(bottom_images, axis=1)
			if len(top_images) == len(bottom_images):
				images = np.concatenate((np.concatenate(top_images, axis = 1), images), axis = 0)
			cv2.imwrite(write_path, images)
		except:
			print(write_path, ': Could not write')

def input_to_seg(folder_name):
	return

# def create_images(folder_name, current_step, results_folder, padding=100, save=True):
# 	files = os.listdir(os.path.join('data/images', folder_name))
# 	files.sort()
# 	#methods = ['contour', 'gradient', 'generic', 'random', 'quadrilaterals']
# 	#if current_step in methods:
# #		opt = EM(folder_name, init_method=current_step, padding=padding)
# #	else:
# 	opt = EM(folder_name, padding=padding)
# 	pbar = tqdm(files)
# 	for file_name in pbar:
# 		pbar.set_description("Processing %s" % file_name)#Show miou
# 		polygons, face_labels = load_polygons(file_name, folder_name, 'quadrilaterals', results_folder)



def cuboid_optimization(folder_name, current_step, results_folder, save=True, padding=100):
	files = os.listdir(os.path.join('data/images', folder_name))
	files.sort()
	init_methods = ['contour', 'gradient', 'generic', 'random', 'quadrilaterals']
	# if current_step in methods:
		# opt = EM(folder_name, init_method=current_step, padding=padding)
	# else:
	opt = EM(folder_name, init_method=current_step, padding=padding)
	evaluator = Metrics(5)
	pck_list = []
	results_dict = {}

	print(folder_name, current_step)
	pbar = tqdm(files)
	for file_name in pbar:

		# if len(results_dict) > 5:
		# 	continue
		results_dict[file_name] = {}

		#if file_name != '854.png':
		#	continue

		pbar.set_description("Processing %s" % file_name)#Show miou
		prob_map = opt.get_numpy_array(file_name + '.npy')
		#if previous_step in ['quadrilaterals', 'merged', 'combined_regular', 'combined_vp']:
			#polygons, face_labels = load_polygons(file_name, folder_name, previous_step)

		t0 = time.time()
		if current_step in init_methods:#== 'quadrilaterals':
			polygons,face_labels,init_polygons,init_face_labels = opt.find_quadrilaterals(prob_map, return_initial_polygon=True)
			if save:
				pred = opt.get_segmentation_output(init_polygons, init_face_labels)
				pred = np.expand_dims(pred, axis=0)
				save_polygons_and_image(init_polygons, init_face_labels, pred, folder_name, file_name, current_step + '_before', opt, results_folder)
		elif current_step == 'merged':
			polygons, face_labels = load_polygons(os.path.join('data', results_folder, 'quadrilaterals', folder_name, file_name))
			polygons,face_labels = opt.combine_quadrilaterals(prob_map, polygons, face_labels)
		elif current_step == 'cuboid_regular':
			polygons, face_labels = load_polygons(os.path.join('data', results_folder, 'merged', folder_name, file_name))
			polygons,face_labels = opt.opt_combined_regular(prob_map, polygons, face_labels)
		elif current_step == 'cuboid_vp':
			#opt.set_images(file_name, set_gt=False)
			polygons, face_labels = load_polygons(os.path.join('data', results_folder, 'merged', folder_name, file_name))
			polygons = opt.opt_vp(polygons, face_labels, prob_map)
		elif current_step == 'cuboid_post':
			#opt.set_images(file_name, set_gt=False)
			polygons, face_labels = load_polygons(os.path.join('data', results_folder, 'cuboid_regular', folder_name, file_name))
			polygons = opt.opt_vp(polygons, face_labels, prob_map)

		t1 = time.time()
		evaluator.add_time(t1-t0)

		if polygons is None:
			print('Polygons is None for file:', file_name)

		prediction_map = add_to_evaluator(opt, evaluator, polygons, face_labels, folder_name, file_name)
		if save:
			save_polygons_and_image(polygons, face_labels, prediction_map, folder_name, file_name, current_step, opt, results_folder)

		if current_step not in init_methods:
			pck_list = calculate_pck(polygons, face_labels, file_name, folder_name, results_folder, padding=opt.padding)

		results_dict[file_name]['accuracy'] = evaluator.Pixel_Accuracy()
		results_dict[file_name]['miou'] = evaluator.Mean_Intersection_over_Union()
		results_dict[file_name]['time'] = float(sum(evaluator.time)/len(evaluator.time))
		results_dict[file_name]['pck'] = pck_list
		evaluator.reset()
		pck_list = []

	acc = [results_dict[x]['accuracy'] for x in results_dict]#evaluator.Pixel_Accuracy()
	acc_mean = statistics.mean(acc)
	acc_std = statistics.stdev(acc)
	print('acc', acc_mean, acc_std)
	miou = [results_dict[x]['miou'] for x in results_dict]#evaluator.Pixel_Accuracy()
	miou_mean = statistics.mean(miou)
	miou_std = statistics.stdev(miou)
	print('miou', miou_mean, miou_std)
	timing = [results_dict[x]['time'] for x in results_dict]#evaluator.Pixel_Accuracy()
	timing_mean = statistics.mean(timing)
	timing_std = statistics.stdev(timing)
	print('time', timing_mean, timing_std)
	for x in results_dict:
		pck_list.extend(results_dict[x]['pck']) 
	if len(pck_list) != 0:
		#print(pck_list)
		pck_mean = statistics.mean(pck_list)#float(sum(pck_list)/len(pck_list))
		pck_std = statistics.stdev(pck_list)#float(sum(pck_list)/len(pck_list))
		print('pck', pck_mean, pck_std)
	else:
		pck_mean = 0
		pck_std = 0

	# print('accuracy:', accuracy)
	# print('miou:', miou)
	# print('pck:', pck)
	# print('average time:', av_time)

	if save:
		with open(os.path.join('data', results_folder, 'stats', 'metrics'), 'a') as f:
			write_string = '{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\n'.format(
				folder_name, current_step, acc_mean, acc_std, miou_mean, miou_std, pck_mean, pck_std, len(pck_list), timing_mean, timing_std)
			f.write(write_string)

		with open(os.path.join('data', results_folder, 'stats', folder_name + '_' + current_step + '.json'), 'w+') as f:
			json.dump(results_dict, f)

if __name__=="__main__":

	#Maybe do all experiments in one go, just change the paramaters

	rs_folders = ['results_init']
	folders = ['MVI_3015.MP4', 'MVI_4627.MP4', 'MVI_3018.MP4']
	# folder_name = 'MVI_3015.MP4'
	methods = ['random', 'generic', 'contour', 'gradient', 'quadrilaterals']
	#methods = ['contour']
	for folder_name in folders:
		for method in methods:
			for results_folder in rs_folders:
				padding = 0 if results_folder == 'results_0' else 100
				#cuboid_optimization(folder_name, method, results_folder, padding=padding)
				#show_image(folder_name, method, results_folder)
				#show_image(folder_name, method + '_before', results_folder)
				make_complete_image(folder_name, results_folder, methods)
			break

	# rs_folders = ['results_100', 'results_0']
	# folders = ['MVI_3015.MP4', 'MVI_4627.MP4', 'MVI_3018.MP4']
	# # folders = ['MVI_4627.MP4']
	# methods = ['quadrilaterals', 'merged', 'cuboid_regular', 'cuboid_vp', 'cuboid_post']
	# #methods = ['cuboid_post']
	# #methods = ['DL_segmentation']
	# for folder_name in folders:
	# 	#for method in methods:
	# 	for results_folder in rs_folders:
	# 		padding = 0 if results_folder == 'results_0' else 100
	# 			#cuboid_optimization(folder_name, method, results_folder, padding=padding)
	# 		make_complete_image(folder_name, results_folder, methods)

	# 			#pck_for_files(folder_name, method, results_folder, padding=padding)
	# 			#DL_segmentation_acc(folder_name, method, results_folder)
	# 			#show_image(folder_name, method, results_folder, padding=padding)
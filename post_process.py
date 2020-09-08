from tqdm import tqdm
import os
import numpy as np
import time

from homography.EM import EM
from homography.homography import Homography
from homography.tag_points import Tag
from project_code.saver.saver import ImgSaver
from project_code.saver.saver import ArraySaver
from project_code.dataloader.utils import decode_seg_map_sequence
from project_code.metrics.metrics import Metrics

def get_polygon_array(name, vid_name):
	path = os.path.join('data/results/polygons', vid_name, name)
	with open(path, 'rb') as f:
		array0 = np.load(f)
		array1 = np.load(f)
	return array0, array1

def vp_pipe(folder_name):
	post = EM('data', folder_name)

	files = os.listdir(os.path.join('data/images', folder_name))
	#files = ['233.png', '242.png', '251.png', '434.png', '863.png']
	for file in tqdm(files):

		#file = '66.png'

		post.set_images(file)

		polygons,face_labels = get_polygon_array(file + '.npy', folder_name)
		array = post.get_numpy_array(file + '.npy')
		polygons = post.opt_vp(polygons, face_labels, array, show=True, filepath=os.path.join(folder_name, file))

		pred_post = post.get_segmentation_output(polygons, face_labels)

		pred_post = np.expand_dims(pred_post, axis=0)
		pred_post = decode_seg_map_sequence(pred_post)

		img_saver = ImgSaver()
		img_saver(pred_post[0], file, folder=os.path.join('segmentation_vp', folder_name))

def EM_pipe(folder_name):
	files = os.listdir(os.path.join('data/images', folder_name))

	files = ['510.png']#, '242.png', '251.png', '434.png', '863.png']

	for file in files:
		print(file)
		post = EM('data', folder_name)
		post.set_images(file)

		array = post.get_numpy_array(file + '.npy')

		polygons, face_labels = post.EM(array)

		pred_post = post.get_segmentation_output(polygons, face_labels)

		pred_post = np.expand_dims(pred_post, axis=0)
		pred_post = decode_seg_map_sequence(pred_post)

		img_saver = ImgSaver()
		array_saver = ArraySaver()
		img_saver(pred_post[0], file, folder=os.path.join('segmentation_post', folder_name))
		array_saver(polygons, file, folder=os.path.join('polygons', folder_name), array1=face_labels)

def tag_pipe(folder_name):
	tagged_points_dict = {}
	files = os.listdir(os.path.join('data/images', folder_name))
	for file in files:
		polygons,face_labels = get_polygon_array(file + '.npy', folder_name)
		tagger = Tag()
		points, labels = tagger(polygons,face_labels)
		ratio_scores = tagger.compute_ratios(points,labels)
		tagged_points_dict[file] = {'points': points, 'labels': labels, 'ratio_scores': ratio_scores}
		tagger.show_points(points, file, folder_name, write=True)

	tagged_points_dict = logically_order_tags(tagged_points_dict)
	return tagged_points_dict

def best_ratio_score_imgs(tagged_points_dict):
	best_scores = {'Front': {}, 'Back': {}, 'Left': {}, 'Right': {}, 'Top': {}}

	for img in tagged_points_dict:
		labels = tagged_points_dict[img]['labels']
		ratio_scores = tagged_points_dict[img]['ratio_scores']
		for ratio_score,label in zip(ratio_scores,labels):
			if 'score' not in best_scores[label] or ratio_score < best_scores[label]['score']:
				best_scores[label] = {'img': img, 'score': ratio_score}
	print(best_scores)

def logically_order_tags(tagged_points_dict):
	'''
	Method that corrects wrong single visible side front or back predictions
	'''

	def sort_keys(elem):
		name = elem.split('.')[0]
		return int(name)

	tagger = Tag()
	reverse = False
	while True:
		keys = tagged_points_dict.keys()
		keys = sorted(keys, key = sort_keys, reverse=reverse)

		previous_labels = None
		for img_name in keys:
			labels = tagged_points_dict[img_name]['labels']
			if len(labels) == 1 and labels[0] == 'Front':
				if previous_labels is None:
					continue
				if 'Front' not in previous_labels:
					tagged_points_dict[img_name]['labels'] = ['Back']
					tagged_points_dict[img_name]['points'] = tagger.swap_tags(tagged_points_dict[img_name]['points'])

			previous_labels = tagged_points_dict[img_name]['labels']
		if reverse:
			break
		else:
			reverse = True
	return tagged_points_dict

def homography_pipe(folder_name, tagged_points_dict):
	files = os.listdir(os.path.join('data/images', folder_name))
	for file in files:
		points, labels = tagged_points_dict[file]['points'], tagged_points_dict[file]['labels']
		homography = Homography()
		imgs = {}
		for side in labels:
			M = homography.calculate_homography(points, side)
			if M is not None:
				img = homography.apply_homography(M, file, folder_name, side)
				imgs[side] = img
				homography.show_homography(img, file, folder_name, side, write=True)
		flat_img = homography.fold_out_container(imgs)
		homography.show_homography(flat_img,file,folder_name,'folded_out', write=True)

def make_video_pipe(folder_name):
	homography = Homography()
	homography.show_warped_video(folder_name)

if __name__=="__main__":
	#np.random.seed(41)
	folder_name = 'MVI_3015.MP4'
	print('Running post processing on the video', folder_name)

	# #print('Running EM')
	EM_pipe(folder_name)

	# print('Tagging points')
	# tagged_points_dict = tag_pipe(folder_name)

	# print('Finding frames facing the camera')
	# best_ratio_score_imgs(tagged_points_dict)

	# print('Computing homographies')
	# homography_pipe(folder_name, tagged_points_dict)

	# print('Creating video')
	# make_video_pipe(folder_name)


	#vp_pipe(folder_name)
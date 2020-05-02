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

def get_numpy_array(name):
	folder = 'data/results/polygons/MVI_3018.MP4'
	path = os.path.join(folder, name)
	with open(path, 'rb') as f:
		array0 = np.load(f)
		array1 = np.load(f)
	return array0, array1

if __name__=="__main__":
	#post = EM()
	#np.random.seed(41)
	folder_name = 'MVI_3018.MP4'
	path = os.path.join('data/images', folder_name)
	files = os.listdir(path)
	for f in tqdm(files):
		#for i in range(10):

		#f = '346.png'

		#FOR DOING EM
		# array_name = f + 'output.npy'
		# print(array_name)
		# array = post.get_numpy_array(array_name)
		# post.set_images(f)

		# # t0 = time.time()
		# polygons, face_labels = post.EM(array)
		# # print(time.time() - t0)

		# array_saver = ArraySaver()
		# array_saver(polygons, f + '.npy', folder='polygons', array1=face_labels)

		#FOR SAVING SEG RESULT
		# pred = np.expand_dims(pred, axis=0)
		# pred = decode_seg_map_sequence(pred)

		# img_saver = ImgSaver()
		# img_saver(pred[0], f, folder='yoyoyo')


		#FOR TAGGING
		polygons,face_labels = get_numpy_array(f + '.npy')
		tagger = Tag()
		points, labels = tagger(polygons,face_labels)
		tagger.show_points(points, f, folder_name, write=True)

		homography = Homography()
		for side in labels:
			M = homography.calculate_homography(points, side)
			#M = np.linalg.inv(M)
			#print('M', M)
			if M is not None:
				homography.show_homography(M, f, folder_name, side, write=True)
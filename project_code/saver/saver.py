import os
import torch
from PIL import Image
import numpy as np

class ModelSaver():
	def __init__(self, options):
		self.model_path = os.path.join(options['model']['save_path'],
			options['model']['checkpoint'])

	def save_model(self, model, epoch, best_acc, scheduler, optimizer):
		print("Saving model")
		state = {
			"epoch": epoch,
			"model_state": model.state_dict(),
			"best_acc": best_acc,
			"scheduler": scheduler,
			"optimizer": optimizer,
		}
		torch.save(state, self.model_path)

	def load_model(self, model):
		if os.path.isfile(self.model_path):
			checkpoint = torch.load(self.model_path)
			model.load_state_dict(checkpoint["model_state"])
			start_epoch = checkpoint["epoch"]
			best_acc = checkpoint["best_acc"]
			scheduler = checkpoint["scheduler"]
			optimizer = checkpoint["optimizer"]
			print(
				"Loaded checkpoint '{}' (iter {})".format(
				self.model_path, checkpoint["epoch"]
				)
			)
		else:
			print("No checkpoint found at '{}'".format(self.model_path))
			start_epoch = 0
			best_acc = 0
			scheduler = None
			optimizer = None
		return model, start_epoch, best_acc, scheduler, optimizer

class ImgSaver():
	def __init__(self, results_folder='results'):
		self.save_folder = os.path.join('data', results_folder)

	def __call__(self, img, name, img2=None, folder=None):
		img = np.array(img).astype(np.uint8)
		if img.shape[0] == 3:
			img = np.transpose(img, (1,2,0))
		if img2 is not None:
			img2 = np.array(img2).astype(np.uint8)
			if img2.shape[0] == 3:
				img2 = np.transpose(img2, (1,2,0))
			img = np.hstack((img,img2))
		img = Image.fromarray(img)

		if folder is not None:
			folder_path = os.path.join(self.save_folder, folder)
			if not os.path.isdir(folder_path):
				os.mkdir(folder_path)
			full_path = os.path.join(folder_path, name)
		else:
			full_path = os.path.join(self.save_folder, name)
		img.save(full_path)


class ArraySaver():
	def __init__(self, results_folder='results'):
		self.save_folder = os.path.join('data', results_folder)

	def __call__(self, array, name, folder=None, array1=None):
		if folder is not None:
			folder_path = os.path.join(self.save_folder, folder)
			if not os.path.isdir(folder_path):
				os.mkdir(folder_path)
			full_path = os.path.join(folder_path, name)
		else:
			full_path = os.path.join(self.save_folder, name)
		with open(full_path, 'wb') as f:
			array = np.asarray(array)
			np.save(f, array)
			if array1 is not None:
				array1 = np.asarray(array1)
				np.save(f, array1)
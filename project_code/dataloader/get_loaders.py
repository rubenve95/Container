from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Subset
import numpy as np

from project_code.dataloader.dataloader.loader import Loader

class LoaderGetter():
	def __init__(self, options, specific_folders={}, shuffle_dataset=True):
		self.dataset = Loader(options)
		self.batch_size = options['training']['batch_size']
		self.shuffle_dataset = shuffle_dataset

		self.dataset_size = len(self.dataset)
		indices = list(range(self.dataset_size))
		self.indices_split = {}

		#if specified, set a data split equal to the content of a folder
		for split in specific_folders:
			split_indices = self.dataset.get_folder_indices(specific_folders[split])
			self.indices_split[split] = split_indices
			assert split_indices != []
			assert all(elem in indices for elem in split_indices)
			indices = list(set(indices).difference(split_indices))

		if shuffle_dataset:
			np.random.shuffle(indices)

		for split in ['val', 'test']:
			if split not in specific_folders:
				split_point = int(np.floor(options['data']['split'][split] * self.dataset_size))
				split_indices = indices[:split_point]
				self.indices_split[split] = split_indices
				indices = list(set(indices).difference(split_indices))

		if 'train' not in specific_folders:
			self.indices_split['train'] = indices

		self.loaders = {}

	def update_dataloaders(self, update_splits=['train','val','test']):
		if self.shuffle_dataset and 'train' in update_splits:
			np.random.shuffle(self.indices_split['train'])
		for split in update_splits:
			batch_size = self.batch_size if split == 'train' else 1
			subset = Subset(self.dataset, self.indices_split[split])
			self.loaders[split] = DataLoader(subset, batch_size=batch_size,
			 									num_workers=0, shuffle=False)

	def __call__(self, split, update=True):
		if update:
			self.update_dataloaders(update_splits=[split])
		self.dataset.split = split
		return self.loaders[split]

	def get_size(self, split='all'):
		if split == 'all':
			return sum([len(self.indices_split[s]) for s in self.indices_split])
		return len(self.indices_split[split])
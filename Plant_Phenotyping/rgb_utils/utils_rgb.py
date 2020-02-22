import torch
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit
from PIL import Image


dai_dict = dict({
	'Z': {
		'dai_offset': 9,
		1: {
			1: -1, 2: -1, 3: -1,
			4: 9, 5: 9, 6: 9, 7: 9, 8: 9,
			9: 14, 10: 14, 11: 14, 12: 14, 13: 14,
			14: 19, 15: 19, 16: 19, 17: 19, 18: 19,
		},
	},
})


# mapping the labels of incomplete dataset
dai_incom_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 9: 8, 12: 9, 13: 10, 14: 11}
dai_incom_dict_inv = {value: key for key, value in dai_incom_dict.items()}


for dai in range(2, 6):
	dai_dict['Z'][dai] = dai_dict['Z'][1].copy()
	for i in dai_dict['Z'][dai].keys():
		if type(i) == int and i >= 4:
			dai_dict['Z'][dai][i] = dai_dict['Z'][dai][i] + dai - 1


def get_dai_label(sample_id):
	"""
	get day after incubation given a string of the sample ID.
	sample_id e.g. '1,Z12,...'
	"""
	sample_id = sample_id.split(",")
	# sample_id e.g. '1,Z12,...'
	day = sample_id[0]
	plant_type = sample_id[1][0]
	sample_num = sample_id[1][1:]
	label = dai_dict[plant_type][int(day)][int(sample_num)]
	if label == -1:
		return 0
	else:
		return label + 1 - dai_dict[plant_type]['dai_offset']


class CustomRGBDataset(Dataset):
	"""
	A custom Tensordataset class for rgb images.

	Args:
		tensors:    a tuple containing the x (input) and possibly y (target) data,
					e.g. tensors = (x) or tensors = (x, y).
		rescale_size:   scalar, if it is the wish to rescale the individual input
						data samples. rescale_size is used for the height as well as
						the width of the image.
	"""

	# def __init__(self, tensors, batch_size, rescale_size=None):
	def __init__(self, tensors, rescale_size=None):

		assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
		# boolean indicating whether target (y) data has been passed
		self.target = True
		if not isinstance(tensors, tuple):
			tensors = (tensors, None)
			self.target = False

		# boolean indicating whether an image mask has been added
		self.mask = False
		if len(tensors) == 3:
			self.mask = True

		self.tensors = tensors
		self.rescale_size = rescale_size

	def __getitem__(self, index):
		# batch_idx = (np.ceil(index/self.batch_size) +
		#             1*(np.mod(index, self.batch_size) == 0))

		x = self.tensors[0][index]

		if self.rescale_size is not None:
			x = self.resize(x)

		if self.target:
			y = self.tensors[1][index]
			if self.mask:
				a = self.tensors[2][index]
				return (x, a), y
			else:
				return (x, torch.tensor([])), y
		else:
			return x

	def resize(self, x):
		x_resize = np.zeros((x.shape[0], self.rescale_size, self.rescale_size))
		for ch in np.arange(0, x.shape[0]):
			x_resize[ch, :, :] = cv2.resize(x.numpy()[ch, :, :],
											(self.rescale_size, self.rescale_size),
											interpolation=cv2.INTER_AREA
											)
		return torch.tensor(x_resize)

	def __len__(self):
		return self.tensors[0].size(0)


def imread_from_fp_rescale_rotate_flatten(fplist, y=np.array([]), rescale_size=None, n_rot_per_img=0):
	"""
	Load files from list of file paths, fplist) and create X matrix, where each
	row is a sample, each column a feature, i.e. pixel in r, g, or b channel,
	i.e. X : [nsamples x (256 x 256 x 3)].
	If desired rescale the images and randomly rotate these n_rot_per_img
	times. n_rot_per_img=0 means no rotation and each image is read as is in
	original. n_rot_per_img>0 means every image will be return rotated n_rot_per_img
	times, without keeping the original image rotation.
	If labels are given in array y, these will be shuffled as the X array.
	"""
	if n_rot_per_img == 0:
		rot = False
		# just a hack
		n_rot_per_img = 1
		print("FYI: Running without rotations ...")
	else:
		rot = True
		print("FYI: Running with {} rotations per image ...".format(n_rot_per_img))
		# need to update label array if rotations are desired
		if np.any(y):
			y = np.repeat(y, n_rot_per_img)

	if rescale_size is not None:
		data_flatten = np.zeros((fplist.size * n_rot_per_img, rescale_size * rescale_size * 3))
	else:
		ex_img = Image.open(fplist[0])
		data_flatten = np.zeros((fplist.size * n_rot_per_img, ex_img.size[1] * ex_img.size[0] * 3))

	# load and preprocess each individual image
	for idx in np.arange(fplist.size * n_rot_per_img):
		fplist_idx = int(idx / n_rot_per_img)
		img = Image.open(fplist[fplist_idx])

		if rescale_size is not None:
			img = img.resize((rescale_size, rescale_size), Image.ANTIALIAS)

		if rot:
			img = img.rotate(np.random.randint(0, 360))

		data_flatten[idx, :] = np.array(img).flatten(order='C')

	# final shuffle along first dim especially needed for when rotation is used
	perm_ids = np.random.permutation(np.arange(0, data_flatten.shape[0]))
	if y.size is not 0:
		return fplist[perm_ids], data_flatten[perm_ids], y[perm_ids], perm_ids
	else:
		return fplist[perm_ids], data_flatten[perm_ids], perm_ids


def imread_from_fp_rescale_rotate_flatten_returnmasks(fplist, fp_mask, y=np.array([]), rescale_size=None,
													  n_rot_per_img=0, rrr=False, model='vgg'):
	"""
	A very hacky solution to also loading the masks!
	Load files from list of file paths, fplist) and create X matrix, where each
	row is a sample, each column a feature, i.e. pixel in r, g, or b channel,
	i.e. X : [nsamples x (256 x 256 x 3)].
	If desired rescale the images and randomly rotate these n_rot_per_img
	times. n_rot_per_img=0 means no rotation and each image is read as is in
	original. n_rot_per_img>0 means every image will be return rotated n_rot_per_img
	times, without keeping the original image rotation.
	If labels are given in array y, these will be shuffled as the X array.
	"""
	if n_rot_per_img == 0:
		rot = False
		# just a hack
		n_rot_per_img = 1
		print("FYI: Running without rotations ...")
	else:
		rot = True
		print("FYI: Running with {} rotations per image ...".format(n_rot_per_img))
		# need to update label array if rotations are desired
		if np.any(y):
			y = np.repeat(y, n_rot_per_img)

	if rescale_size is not None:
		data_flatten = np.zeros((fplist.size * n_rot_per_img, rescale_size * rescale_size * 3))
		# masks_flatten = np.zeros((fplist.size * n_rot_per_img, rescale_size * rescale_size))
		if model == 'vgg':
			masks_flatten = np.zeros((fplist.size * n_rot_per_img, 14*14))
	else:
		ex_img = Image.open(fplist[0])
		data_flatten = np.zeros((fplist.size * n_rot_per_img, ex_img.size[1] * ex_img.size[0] * 3))
		masks_flatten = np.zeros((fplist.size * n_rot_per_img, ex_img.size[1] * ex_img.size[0]))

	# load the mask dictionary
	with open(fp_mask, 'rb') as handle:
		mask_dict = pickle.load(handle)

	# load and preprocess each individual image
	for idx in np.arange(fplist.size * n_rot_per_img):
		fplist_idx = int(idx / n_rot_per_img)
		img = Image.open(fplist[fplist_idx])

		img = np.array(img)
		key = fplist[fplist_idx].split('/')[-1].split('.')[0]
		mask = mask_dict[key]
		# if to train on masked images
		if not rrr:
			print("FYI: The images are being cut out!!!, the masks are being returned as well")
			# loop over rgb channels
			for i in np.arange(0, 3):
				img[mask==0, i] = 0
		img = Image.fromarray(img) 	# very hacky, pil objects do not support item assignment
		mask = img_frombytes(mask)

		if rescale_size is not None:
			img = img.resize((rescale_size, rescale_size), Image.ANTIALIAS)
			if model == 'vgg':
				mask = mask.resize((14, 14), Image.ANTIALIAS)

		if rot:
			rot_degree = np.random.randint(0, 360)
			img = img.rotate(rot_degree)
			mask = mask.rotate(rot_degree)

		# invert mask as rrr loss need a mask where it is 0 in that region where the model should focus
		mask = np.logical_not(mask)

		data_flatten[idx, :] = np.array(img).flatten(order='C')
		masks_flatten[idx, :] = np.array(mask).flatten(order='C')

	# final shuffle along first dim especially needed for when rotation is used
	perm_ids = np.random.permutation(np.arange(0, data_flatten.shape[0]))
	return fplist[perm_ids], data_flatten[perm_ids], masks_flatten[perm_ids], y[perm_ids], perm_ids


# Use .frombytes instead of .fromarray.
# This is >2x faster than img_grey
def img_frombytes(data):
	"""
	See https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
	:param data:
	:return:
	"""
	size = data.shape[::-1]
	databytes = np.packbits(data, axis=1)
	return Image.frombytes(mode='1', size=size, data=databytes)


def reshape_flattened_to_tensor_rgb(data, width_height):
	"""
	Reshape X (flattened data) to original size and swap axis to be conform with
	pytorch rgb tensors.
	"""
	data = np.reshape(data, (data.shape[0], width_height, width_height, 3))
	data = np.swapaxes(data, 1, 3)
	data = np.swapaxes(data, 2, 3)
	return data

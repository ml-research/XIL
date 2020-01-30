"""
This file contains code for performing SpRAy anaylsis using CAM heatmaps rather than LRP heatmaps.
Here the strategies are clustered using only the heatmaps where ypred == camclass, i.e. the activation map that was used
to decide for a class.
@author: Wolfgang Stammer
@date: 26.08.2019

Usage example:

python create_final_plots.py --mask --rrr --l2-grads 8 --norm --image-type 'rgb' --metric 'cityblock'

"""
# disable multithreading
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import pickle
import argparse
import shutil
import seaborn as sns
import errno
import os
from scipy.sparse import csr_matrix
from skimage.transform import downscale_local_mean
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from random import sample
from PIL import Image, ImageEnhance
from scipy import fftpack
from skimage import feature
from sklearn import preprocessing
from scipy import ndimage as ndi
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import rc

sys.path.insert(0, 'libs')
from auto_spectral_clustering.autosp import predict_k
from utils import get_dai_label

sns.set(style='ticks', palette='Set2')
sns.despine()
rc('text', usetex=True)
#----------------------------------------------------------------------------------------------------------------------#

seed = 42
np.random.seed(seed)

FP_ORIG_IMGS = "/home/schramowski/datasets/deepplant/data/parsed_data/Z/VNIR/rgb_new/"

FACTOR_DOWNSCALE = 5
NUM_SUBPLOT_ITEMS = 8
NJOBS = 4

# as in Lapushkin et al., 2019 supplementary section 6.2
PERPLEXITY = 7
EARLY_EXAGGERATION = 6
# as in Lapushkin et al., 2019 supplementary section 6.1 that is < 1
EPS = 0.05

CMAP = 'viridis'
#----------------------------------------------------------------------------------------------------------------------#


def visualize_scatter_with_images_clusters(X_2d_data, images, fp, cluster_ids, figsize=(45,45), image_zoom=1):
	"""
	Creates a scatter plot with images plotted rather than
	points. Around each image is a space colored by the cluster.

	:param X_2d_data: [n x 2], 2D embedding of images, e.g. t-SNE
	:param images: [n x w x h], 3D matrix containing the individual images
	:param fp: file path so save to
	:param figsize: tuple, size of figure
	:param image_zoom: amount of zoom of the images, positive float

	:return:
	"""
	fig, ax = plt.subplots(figsize=figsize)
	k = np.unique(cluster_ids).shape[0]
	for i in np.arange(0, k):
		ids = np.where((cluster_ids == i))
		scatter = plt.scatter(X_2d_data[ids, 0], X_2d_data[ids, 1], s=20000, alpha=0.8)
	plt.legend(['cluster ' + str(i) for i in np.arange(k)], prop={'size': 40})

	artists = []
	for xy, i in zip(X_2d_data, images):
		x0, y0 = xy
		img = OffsetImage(i, zoom=image_zoom, cmap=CMAP)
		ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
		artists.append(ax.add_artist(ab))
	ax.update_datalim(X_2d_data)
	ax.autoscale()
	ax.axis('off')
	fig.savefig(fp, bbox_inches='tight')
	plt.close()


def load_data(filenames, image_type):
	"""
	Load the cams specified in filenames, create an info_array storing information to each cam
	(cam_class, dai, gt_label, pred_label), downscale the cams.

	:param filenames: numpy array of filepaths
	:param image_type: string, either 'rgb' or 'hs'

	:return:
	infoarray: [n x 4], array storing information to each sample (cam_class, dai, gt_label, pred_label)
	cams: [n x w x h], array with all cams
	cams_flatten: [n x (w * h)], flattened array of all cams
	cams_downscaled: [n x (w*1/FACTOR_DOWNSCALE) x (h*1/FACTOR_DOWNSCALE)], downscaled cams
	cams_downscaled_flatten: [n x ((w*1/FACTOR_DOWNSCALE) * (h*1/FACTOR_DOWNSCALE))], flattened downscaled cams
	"""
	# contains the cam class id, dai, true label, predicted label for each file in filenames
	info_array = np.zeros((len(filenames), 4))
	cams = []
	cams_downscaled = []
	cams_mean = []
	cams_mean_downscaled = []
	for idx in np.arange(0, len(filenames)):
		fname = filenames[idx]
		tmp = fname.split('/')[-1].split('.')[0].split('_')
		# cam_class
		info_array[idx, 0] = int(tmp[6])
		# dai
		info_array[idx, 1] = get_dai_label(fname.split(".")[0].split("/")[-1].replace("_", ","))
		# true label
		info_array[idx, 2] = int(tmp[8])
		# predicted label
		info_array[idx, 3] = int(tmp[10])

		if 'rgb' in image_type:
			cam = np.load(fname)

			# downscale for computational reasons when using clustering
			cams_downscaled.append(downscale_local_mean(cam, (FACTOR_DOWNSCALE, FACTOR_DOWNSCALE)))
			cams.append(cam)
		elif 'hs' in image_type:
			with open(fname, 'rb') as handle:
				cam_dict = pickle.load(handle)
			cam = cam_dict["cams"]
			cam_mean = cam_dict["spatial_cams"]
			spectral_cams = cam_dict["spectral_cam"]

			cam_mean = np.maximum(cam_mean, 0)
			cam_mean = (cam_mean - np.min(cam_mean)) / (np.max(cam_mean) - np.min(cam_mean))  # Normalize between 0-1
			cam_mean = np.uint8(cam_mean * 255)  # Scale between 0-255 to visualize

			# downscale for computational reasons when using clustering
			cams_downscaled.append([downscale_local_mean(cam[i, :, :], (FACTOR_DOWNSCALE, FACTOR_DOWNSCALE))
									for i in range(cam.shape[0])])
			cams.append(cam)
			cams_mean.append(cam_mean)
			cams_mean_downscaled.append(downscale_local_mean(cam_mean, (FACTOR_DOWNSCALE, FACTOR_DOWNSCALE)))

	cams = np.array(cams)
	cams_mean = np.array(cams_mean)
	cams_mean_downscaled = np.array(cams_mean_downscaled)
	cams_downscaled = np.array(cams_downscaled)

	if 'rgb' in image_type:
		cams_flatten = np.reshape(cams, (cams.shape[0], cams.shape[1] * cams.shape[2]))
		cams_downscaled_flatten = np.reshape(cams_downscaled, (cams_downscaled.shape[0],
															   cams_downscaled.shape[1] * cams_downscaled.shape[2]))
	elif 'hs' in image_type:
		cams_flatten = np.reshape(cams, (cams.shape[0], cams.shape[1], cams.shape[2] * cams.shape[3]))
		cams_downscaled_flatten = np.reshape(cams_downscaled, (cams_downscaled.shape[0], cams_downscaled.shape[1],
															   cams_downscaled.shape[2] * cams_downscaled.shape[3]))

	return info_array, cams, cams_flatten, cams_mean, cams_mean_downscaled, cams_downscaled, cams_downscaled_flatten


def concatenate_list_data(list):
	"""
	Helper function turning a list of elements into a string.

	:param list: list of elements

	:return concatenated string of all elements in list
	"""
	result= ''
	for element in list:
		result += str(element)
	return result


def perform_fourier(X, image_type):
	"""
	Performs Fourier transfrom on every image in X. For hs images the fourier of the image of every channel is computed.

	:param X: [n x w x h], array of n imgs, each w x h large
	:param image_type: string, either 'rgb' or 'hs'

	:return list of fourier transformed images
	"""
	res = []

	if 'rgb' in image_type:
		for img_id in np.arange(0, X.shape[0]):
			ftimage = np.fft.fft2(X[img_id, :, :])
			ftimage = np.fft.fftshift(ftimage)
			res.append(np.abs(ftimage))
	elif 'hs' in image_type:
		for img_id in np.arange(0, X.shape[0]):
			tmp = []
			for channel_id in range(4):
				ftimage = np.fft.fft2(X[img_id, channel_id, :, :])
				ftimage = np.fft.fftshift(ftimage)
				tmp.append(np.abs(ftimage))
			res.append(tmp)
	return np.array(res)


def compute_affinity_matrix(imgs, n_nn, image_type):
	"""
	Computes the affinitiy matrix of the data in imgs, using the k nearest neighbors graph based on the cityblock
	distance.

	:param imgs: [n x w x h], array of n imgs, each w x h large
	:param n_nn: int, number of nearest neighbors
	:param image_type: string, either 'rgb' or 'hs'

	:return affinity matrix of the images
	"""

	# testing code form repo auto_spectral_clustering
	# Calculate connectivity_matrix
	if 'rgb' in image_type:
		fourier_imgs = perform_fourier(imgs, image_type)
		fourier_imgs = np.reshape(fourier_imgs, (imgs.shape[0], imgs.shape[1] * imgs.shape[2]))
		print("Fourier transform computed")
		connectivity = kneighbors_graph(fourier_imgs, metric='cityblock', n_neighbors=n_nn, n_jobs=NJOBS)

	elif 'hs' in image_type:
		fourier_imgs = perform_fourier(imgs, image_type)
		fourier_imgs = np.reshape(fourier_imgs, (imgs.shape[0], imgs.shape[1] * imgs.shape[2] * imgs.shape[3]))
		print("Fourier transform computed")
		connectivity = kneighbors_graph(fourier_imgs, metric='cityblock', n_neighbors=n_nn, n_jobs=NJOBS)

	# as in Von Luxburg, 2007 and Lapushkin et al., 2019
	affinity_matrix = csr_matrix.maximum(connectivity, connectivity.T)

	return affinity_matrix


def get_filenames(image_type, loss_type):
	"""
	Gather those files where the prediction is the same as the cam class

	:param image_type: either 'rgb' or 'hs'
	:param loss_type: either 'default' or 'rrr'

	:return: numpy array of filepaths
	"""
	if 'rgb' in image_type:
		if 'default' in loss_type:
			fp_cams = "path_to_default_rgb_gradcams"
			fp_save = "affinity_matrices/rgb_default_nclusters_affinity.npy"
		elif 'rrr' in loss_type:
			fp_cams = "path_to_rrr_rgb_gradcams"
			fp_save = "affinity_matrices/rgb_rrr_nclusters_affinity.npy"
	elif 'hs' in image_type:
		if 'default' in loss_type:
			fp_cams = "path_to_default_hs_gradcams"
			fp_save = "affinity_matrices/hs_default_nclusters_affinity.npy"
		elif 'rrr' in loss_type:
			fp_cams = "path_to_rrr_hs_gradcams"
			fp_save = "affinity_matrices/hs_rrr_nclusters_affinity.npy"

	filenames = []
	for cam_class in np.arange(0, 2):
		filenames.append(name for name in
						 glob.glob(fp_cams + "*_class_" + str(cam_class) + "_*_ypred_" + str(cam_class) + "*.npy"))
	filenames = np.array([item for sublist in filenames for item in sublist])

	return filenames, fp_save


def compute_save_affinity_matrices():
	"""
	Compute and save the affinity matrix for several configurations of image type and loss type.

	:return: None
	"""
	print("------------------------------------------------")
	# create the directory for the data
	dir_name = "affinity_matrices/"
	try:
		os.makedirs(dir_name)
		print("Creating directory: ", dir_name)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	configs = [['rgb', 'default'], ['rgb', 'rrr'], ['hs', 'default'], ['hs', 'rrr']]
	for config in configs:
		image_type = config[0]
		loss_type = config[1]
		print("Compute affinity matrix for {} and {}".format(image_type, loss_type))
		filenames, fp_save = get_filenames(image_type, loss_type)
		_, _, _, _, _, cams_downscaled, _ = load_data(filenames, image_type)
		print(len(filenames))
		# as in Von Luxburg, 2007
		n_nn = int(np.log(len(filenames)))
		affinity_matrix = compute_affinity_matrix(cams_downscaled, n_nn, image_type)

		# as in Von Luxburg, 2007
		n_nn = int(np.log(len(filenames)))

		# auto_spectral_clustering, get number of clusters
		k, eigenvalues = predict_k(affinity_matrix)

		print("{} clusters estimated".format(k))

		data_dict = {'n_clusters': k, 'affinity': affinity_matrix}

		with open(fp_save, 'wb') as handle:
			pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print("Affinity matrix for {} and {} saved at {}".format(image_type, loss_type, fp_save))


def create_save_image_embedded_tsne_plots():
	"""
	Create the t-sne embedding plots colored by the spectral clustering assignments. This reads the filenames, loads
	data, loads the precomputed affinity matrices, computes the t-sne embedding, performs the clustering and finally
	creates the actual plots colored by the clustering.
	"""
	print("------------------------------------------------")
	# create the directory for the figures
	dir_name = "figures/"
	try:
		os.makedirs(dir_name)
		print("Creating directory: ", dir_name)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	configs = [['rgb', 'default'], ['rgb', 'rrr'], ['hs', 'default'], ['hs', 'rrr']]
	for config in configs:
		image_type = config[0]
		loss_type = config[1]
		filenames, fp_affinity = get_filenames(image_type, loss_type)
		_, cams, _, cams_mean, cams_mean_downscaled, cams_downscaled, _ = load_data(filenames, image_type)
		with open(fp_affinity, 'rb') as handle:
			data_dict = pickle.load(handle)

		k = data_dict['n_clusters']
		affinity_matrix = data_dict['affinity']

		print("Creating plots for {} and {} with {} clusters".format(image_type, loss_type, k))

		# t-SNE embedding 2D
		cams_embedded = TSNE(n_components=2, random_state=seed, metric='precomputed',
							 perplexity=PERPLEXITY,
							 early_exaggeration=EARLY_EXAGGERATION).fit_transform(1 / (affinity_matrix.todense() + EPS))

		print("T-SNE complete")

		# SC clustering
		cl_model = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels="kmeans",
									  random_state=seed, n_jobs=NJOBS).fit(affinity_matrix)

		cl_labels = np.array(cl_model.labels_.astype(np.int))

		print("Clustering complete")

		fp_save_ext = concatenate_list_data(fp_affinity.split('/')[-1].split('.')[0].split('_nclusters_affinity')[0])
		fp_save = "figures/" + fp_save_ext + "_image_embedded_tsne_colored.png"

		if 'rgb' in image_type:
			# scatter plot images instead of points
			visualize_scatter_with_images_clusters(cams_embedded, cams_downscaled,
												   fp=fp_save,
												   cluster_ids=cl_labels, figsize=(45, 45), image_zoom=1.5)
		elif 'hs' in image_type:
			# scatter plot images instead of points
			visualize_scatter_with_images_clusters(cams_embedded, cams_mean_downscaled,
												   fp=fp_save,
												   cluster_ids=cl_labels, figsize=(45, 45), image_zoom=1.5)


def create_embedded_tsne_classificationrate_plots():
	"""
	Creates the embedded tsne plots colored by the model's classification and ground truth labels. This reads the
	filenames, loads data, loads the precomputed affinity matrices, computes the t-sne embedding, performs the
	clustering and finally creates the actual plots colored by the classification and ground truth label.
	"""
	print("------------------------------------------------")
	# create the directory for the figures
	dir_name = "figures/"
	try:
		os.makedirs(dir_name)
		print("Creating directory: ", dir_name)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	# number of clusters found from spectral clustering, must be set manually
	configs = [['rgb', 'default', 5], ['rgb', 'rrr', 2], ['hs', 'default', 2], ['hs', 'rrr', 6]]
	for config in configs:
		image_type = config[0]
		loss_type = config[1]
		filenames, fp_affinity = get_filenames(image_type, loss_type)
		info_array, cams, _, cams_mean, cams_mean_downscaled, cams_downscaled, _ = load_data(filenames, image_type)
		with open(fp_affinity, 'rb') as handle:
			data_dict = pickle.load(handle)

		k = data_dict['n_clusters']
		affinity_matrix = data_dict['affinity']

		print("Creating plots for {} and {} with {} clusters".format(image_type, loss_type, k))

		# t-SNE embedding 2D
		cams_embedded = TSNE(n_components=2, random_state=seed, metric='precomputed',
							 perplexity=PERPLEXITY,
							 early_exaggeration=EARLY_EXAGGERATION).fit_transform(1 / (affinity_matrix.todense() + EPS))

		print("T-SNE complete")

		# plot tsne embedding colored by true and predicted label
		legend_array = ['gt: 0; pred: 0', 'gt: 0; pred: 1', 'gt: 1; pred: 0', 'gt: 1; pred: 1']
		colors = []
		for index in range(4):
			colors.append(list(plt.cm.Set1(index)))
		fig = plt.figure(figsize=(15, 15))
		for gt in np.arange(0, 2):
			for pred in np.arange(0, 2):
				# indices that belong to cluster and to gt and pred
				ids = np.where((info_array[:, 2] == gt) & (info_array[:, 3] == pred))
				plt.scatter(cams_embedded[ids, 0], cams_embedded[ids, 1], s=100, color=colors[gt + (pred * 2)])
		if 'hs' in image_type and 'rrr' in loss_type:
			plt.legend(legend_array, prop={'size': 30}, loc='lower right')
		plt.axis('off')

		fp_save_ext = concatenate_list_data(fp_affinity.split('/')[-1].split('.')[0].split('_affinity')[0])
		fp_save = "figures/" + fp_save_ext + "_tsne_classrates.png"
		fig.savefig(fp_save, bbox_inches='tight')

		plt.close()

#----------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
	compute_save_affinity_matrices()
	# create_save_image_embedded_tsne_plots()
	# create_embedded_tsne_classificationrate_plots()

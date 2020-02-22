"""
This file is for reading and training a pretrained deep neural network from the
pytorch library on the beet (zuckerrueben) data set.

Usage example:

for training with rrr and l2 weight 1:

CUDA_VISIBLE_DEVICES=3 python3 main_rgb.py --lr 0.00005 --batch-size 32 --n-epochs 10 --n-rotations 1 --n-cvruns 5 --cv-run 2 --norm --rrr --l2-grads 1 --train --fp-save path/to/save --fp-data path/to/data

for generating grad cams from a specific model:

CUDA_VISIBLE_DEVICES=3 python3 main_rgb.py --lr 0.00005 --batch-size 32 --n-epochs 10 --n-rotations 1 --n-cvruns 5 --cv-run 2 --norm --rrr --l2-grads 1 --gen-cams --fp-save path/to/save --fp-data path/to/data --cp-fname 'vgg_cvnum_2_epoch_1_evalbalacc_5000_trainraloss_8.8632_trainrrrloss_3.9064_besttestacc.pth'

"""
# disable multithreading
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import glob
import argparse
import numpy as np
import os
import sys
import pickle
import errno
import setproctitle
import pandas as pd
import torchvision.models as models
import time
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR, MultiStepLR
from PIL import ImageEnhance
from torch import nn
from skimage import feature
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from rgb_utils.gradcam_rgb import GradCam
from rgb_utils.misc_functions_rgb import save_class_activation_images
from rgb_utils.rrr_loss_rgb import rrr_loss_function #RRRLoss
from rgb_utils.utils_rgb import *

np.set_printoptions(precision=3)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)

parser = argparse.ArgumentParser(description='PyTorch CNNs')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for gradient descent (default: 0.0001)')
parser.add_argument('--n-epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--n-rotations', type=int, default=1, metavar='N',
                    help='the number of rotations per image')
parser.add_argument('--n-cvruns', type=int, default=1, metavar='N',
                    help='the number of cross validation splits')
parser.add_argument('--cv-run', type=int, default=1, metavar='N',
                    help='the current cross validation split, must be between 0 and (n-cvruns - 1)')
parser.add_argument('--l2-grads', type=float, default=1, metavar='N',
                    help='regularization parameter for the right reasons part in rrr loss')
parser.add_argument('--rrr', action='store_true', default=False,
                    help='whether to train using the right for right reasons loss')
parser.add_argument('--norm', action='store_true', default=False,
                    help='whether to normalize the data')
parser.add_argument('--train', action='store_true', default=False,
                    help='whether to run the training process on the data')
parser.add_argument('--test', action='store_true', default=False,
                    help='whether to evaluate the performance of a model checkpoint on the test data')
parser.add_argument('--gen-cams', action='store_true', default=False,
                    help='whether to normalize the data')
parser.add_argument('--cp-fname', type=str, default='',
                    help='the name of the model checkpoint')
parser.add_argument('--fp-data', type=str, required=True,
                    help='please specifiy the path to the data folder')
parser.add_argument('--fp-save', type=str, required=True,
                    help='please specifiy the path to the folder in which files will be saved')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.set_num_threads(4)
# ------------------------------------------------------------------------------#


def create_args():
    if args.gen_cams & (args.cp_fname is None):
        raise AssertionError('please specify the name of the model checkpoint file to use for generating the cams!')
    if args.test & (args.cp_fname is None):
        raise AssertionError('please specify the name of the model checkpoint file to use for evaluating!')
    if args.train & args.test:
        raise AssertionError('please specify either train or test, the model is automatically evaluated in the '
                             'training process')

    args.num_classes = 2

    args.model_name = "vgg"

    if args.cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    args.reduction = torch.sum

    if args.norm:
        args.norm_ext = "normalized/"
    else:
        args.norm_ext = "unnormalized/"

    # extension for file saving
    if not args.rrr:
        args.fp_mask = None # if the model is not to be trained using the masks
        args.train_config = "default"
    else:
        # args.fp_mask = os.path.join(args.fp_data, "plant_hs/background_labels/preprocessed_masks.pyu")
        args.fp_mask = "/datasets/dataset_zuckerruebe_Patrick/images_rgb/orig_rgb/preprocessed_masks.pyu"
        if args.l2_grads == 0.1:
            args.train_config = "rrr_l2grads_01"
        else:
            args.train_config = f"rrr_l2grads_{int(args.l2_grads)}"

    args.fp_save = os.path.join(args.fp_save, args.model_name, f"num_classes_{args.num_classes}",
                                args.train_config, args.norm_ext)

    print("\nCurrent argument parameters: \n")
    print("\n".join("{}: {}".format(k, v) for k, v in vars(args).items()))
    # input("\nAre the parameters correct? Press enter to start...\n")


def get_data():
    """
    Loads all filenames of the rgb images, and the relevant train test split, according to args.cv_run.

    :return:
        filenames_allfiles: list of strings, containing the file path to each data sample
        y: list of ints, indicating whether a sample is healthy or sick
    """
    # list of filenames
    filenames_allfiles = np.array([name for name in glob.glob(args.fp_data + "[1-5]/*")])

    # get label list
    y = []
    for fname in filenames_allfiles:
        fname = fname.split(".JPEG")[0].split("/")[-1].replace("_", ",")
        y.append(get_dai_label(fname))
    y = np.array(y)
    y[y > 0] = 1

    return filenames_allfiles, y


def get_data_split(filenames_allfiles):
    """
    Loads the relevant train test split, according to args.cv_run.
    :param filenames_allfiles: list of strings, containing the file path to each data sample
    :return:
        train_index: list of ints, indicating which file name in filenames_allfiles belongs to a training sample
        test_index: list of ints, indicating which file name in filenames_allfiles belongs to a test sample
        train_sample: list of strings, containing the sample id of all training samples
        test_index: list of strings, containing the sample id of all test samples
    """
    # list of train and test sample ids for corresponding cv run
    with open(f"rgb_dataset_splits/train_{args.cv_run}.txt") as f:
        train_samples = f.read().splitlines()
    with open(f"rgb_dataset_splits/test_{args.cv_run}.txt") as f:
        test_samples = f.read().splitlines()

    # find indices of training and test samples given list of all files
    train_index = [i for i, fname in enumerate(filenames_allfiles) if fname.split('/')[-1].split('.')[0] in train_samples]
    test_index = [i for i, fname in enumerate(filenames_allfiles) if fname.split('/')[-1].split('.')[0] in test_samples]

    return train_index, test_index, train_samples, test_samples


def set_parameter_requires_grad(model, feature_extracting):
    """
    Sets the .requires_grad attribute of the parameters in the model to False when
    we are feature extracting. By default, when we load a pretrained model all of
    the parameters have .requires_grad=True, which is fine if we are training from
    scratch or finetuning. However, if we are feature extracting and only want to
    compute gradients for the newly initialized layer then we want all of the other
    parameters to not require gradients.
    Function from:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_model(num_classes, feature_extract=False, use_pretrained=True):
    """
    Choose a pretrained model. Function from:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    :param num_classes: int, the number of classes the model should classify
    :param feature_extract: boolean, whether to only extract the features of the pre-trained
                            model and not further train these, default is False
    :param use_pretrained: boolean, whether to load the pre-trained model weights, default is True
    :return:
        model_ft: the loaded, possibly modified model object which is to be finetuned
        input_size: int, the size of the input of the model
        model_params: list, containing all model parameters
        pau_params: list, optional containing the parameters of the pau layers (default: [])
    """
    # model finetuning
    model_ft = None
    input_size = 0

    if args.model_name == "vgg":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name " + args.model_name + ", exiting...")
        exit()

    model_params = model_ft.parameters()

    return model_ft, input_size, model_params


def train(model, train_loader, criterion, optimizer, class_weights=None, verbose=0):
    """
    Define training function for network
    """

    def save_target_output(self, input, output):
        model.target_output = output

    def forward_pass_on_convolutions(x, target_layer):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        if args.model_name == "vgg":
            for module_pos, module in x.features._modules.items():
                if int(module_pos) == target_layer:
                    module.register_forward_hook(save_target_output)

    model.train()
    forward_pass_on_convolutions(model, target_layer=29)

    current_loss = 0
    current_right_answer_loss = 0
    current_right_reason_loss = 0
    n_batches = np.ceil(len(train_loader.dataset) / args.batch_size)
    all_targets = []
    all_preds = []
    for data, y_train in iter(train_loader):

        x_train = data[0].float()
        masks = data[1].float()
        y_train = y_train.long()

        if args.cuda:
            x_train, masks, y_train = x_train.cuda(), masks.cuda(), y_train.cuda()

        optimizer.zero_grad()

        output = model(x_train)
        _, preds = torch.max(output, 1)

        all_targets += y_train.cpu().numpy().tolist()
        all_preds += preds.cpu().numpy().tolist()

        if not args.rrr:
            loss = criterion(output, y_train)
            right_answer_loss = torch.tensor(0)
            right_reason_loss = torch.tensor(0)
        else:
            loss, right_answer_loss, right_reason_loss = rrr_loss_function(A=masks, X=model.target_output, y=y_train,
                                                                           logits=output, class_weights=class_weights,
                                                                           criterion=criterion, l2_grads=args.l2_grads,
                                                                           reduce_func=args.reduction)

        loss.backward()
        optimizer.step()

        current_loss += loss.item() * x_train.size(0)
        current_right_answer_loss += right_answer_loss.item() * x_train.size(0)
        current_right_reason_loss += right_reason_loss.item() * x_train.size(0)

        if verbose == 1:
            print("Loss: {:6f}; RA Loss: {:6f}; RR Loss: {:6f}".format(loss.item() * x_train.size(0),
                                                                                       right_answer_loss.item()  * x_train.size(0),
                                                                                       right_reason_loss.item()  * x_train.size(0)))

    epoch_loss = current_loss / n_batches
    epoch_acc = balanced_accuracy_score(all_targets, all_preds)

    return epoch_loss, epoch_acc, (current_right_answer_loss / n_batches), (current_right_reason_loss / n_batches)


def test(model, test_loader, criterion, verbose=0):
    """
    Define testing function for network. Returns the average test accuracy over all
    test batches.
    """
    model.eval()
    count = 0

    current_loss = 0
    current_bal_acc = 0
    test_loss = 0
    test_acc = 0
    y_tests = []
    y_preds = []

    n_batches = np.ceil(len(test_loader.dataset) / args.batch_size)

    for data, y_test in iter(test_loader):
        x_test = data[0].float()
        masks = data[1].float()
        y_test = y_test.long()

        if args.cuda:
            x_test, masks, y_test = x_test.cuda(), masks.cuda(), y_test.cuda()

        with torch.no_grad():
            output = model.forward(x_test)
            # Calculate the class probabilities (softmax) for img
            _, y_pred = torch.max(output.data, 1)

            loss = criterion(output, y_test)

            count += 1
            current_loss += loss.item() * x_test.size(0)
            current_bal_acc += balanced_accuracy_score(y_test.cpu(), y_pred.cpu())

            y_tests += y_test.cpu().numpy().tolist()
            y_preds += y_pred.cpu().numpy().tolist()

            if verbose > 0:
                print("Accuracy of network on test images is ... {:.4f}....count: {}".format(
                    balanced_accuracy_score(y_test.cpu(), y_pred.cpu()), count))
                if verbose == 2:
                    print("True labels:\n" + str(y_test.cpu()))
                    print("Predicted labels:\n" + str(y_pred.cpu()))

        test_loss = current_loss / n_batches
        test_acc = current_bal_acc / n_batches

    return test_loss, test_acc


def gen_cams():
    """
    Generate the gradcams of all test data given the filepath to a checkpoint, args.cp_fname.
    """
    # load previous model dictionary
    state_dict = torch.load(args.fp_save + args.cp_fname)
    std_scaler = state_dict['std_scaler']

    # load model
    model, _, model_params = load_model(num_classes=args.num_classes, feature_extract=False, use_pretrained=True)
    model.load_state_dict(state_dict['model_state'], strict=True)
    if args.cuda:
        model.cuda()
    print("The model was loaded from checkpoint")

    filenames_allfiles, y = get_data()

    # load and permutate data
    filenames_allfiles, x_data, y, perm_ids = imread_from_fp_rescale_rotate_flatten(
        fplist=filenames_allfiles, y=y,
        rescale_size=224, n_rot_per_img=0
    )

    # get split from corresponding cv run, based on permutated filenames_allfiles
    train_index, test_index, train_samples, test_samples = get_data_split(filenames_allfiles)

    # only get test data for generating gradcams
    filenames_allfiles = filenames_allfiles[test_index]
    y = y[test_index]
    x_data = x_data[test_index]

    orig_imgs = []
    for fname in filenames_allfiles:
        orig_imgs.append(Image.open(fname).resize((224, 224), Image.ANTIALIAS))

    print("Stdscaler being applied ...")
    x_norm = reshape_flattened_to_tensor_rgb(std_scaler.transform(x_data), width_height=224)

    # create tensors for dataloaders
    tmp_tensors = (torch.tensor(x_norm), torch.tensor(y))

    tmp_dataset = CustomRGBDataset(tensors=tmp_tensors)
    tmp_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=1)

    fp_cvrun = args.fp_save + 'gradcams/cvnum_' + args.cp_fname.split('cvnum_')[-1].split('_')[0] + '/'

    # create the directory of the gradcams in case it does not exist
    for day in np.arange(1, 6):
        try:
            gradcam_day_fp = fp_cvrun + 'day_'+ str(day) + '/'
            os.makedirs(gradcam_day_fp)
            print("Creating directory: ", gradcam_day_fp)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    model.eval()
    for i, (data, y) in enumerate(tmp_loader):
        # convert orig img to edge img
        # Generate noisy image of a square
        im_orig = np.array(orig_imgs[i])[:, :, 1]
        im = preprocessing.scale(np.array(im_orig, dtype=float) / 255)
        im = ndi.gaussian_filter(im, 4)
        # Compute the Canny filter for two values of sigma
        edges1 = feature.canny(im, sigma=3)
        edges1 = Image.fromarray(np.uint8(np.invert(edges1) * 255), 'L')

        enhancer = ImageEnhance.Brightness(orig_imgs[i])
        orig_img = enhancer.enhance(1.8)

        x = data[0].float()

        if args.cuda:
            x = x.cuda()

        with torch.no_grad():
            output = model.forward(x)
            _, y_pred = torch.max(output.data, 1)

        y = y.cpu().numpy().tolist()[0]
        y_pred = y_pred.cpu().numpy().tolist()[0]

        # Grad cam
        grad_cam = GradCam(model, target_layer=29, device=args.device) # last RELU layer of model.features

        # Generate cam mask
        cam_class = y_pred
        print("Pred: {}, Class : {}".format(y_pred, cam_class))
        cam = grad_cam.generate_cam(x, [cam_class])

        tmp = filenames_allfiles[i].split('.')[0].split('/')

        # Save subplots
        f_name = str(tmp[-1]) + '_class_' + str(cam_class)
        fp_cam = os.path.join(fp_cvrun + 'day_' + str(tmp[-2]) + '/',
                          f_name + '_y_' + str(y) + '_ypred_' + str(y_pred) + '_Cam_Heatmap')

        save_class_activation_images(orig_img, cam, fp_cam + '.png',
                                     file_name=f_name,
                                     true_class=y,
                                     pred_class=y_pred, edge_img=edges1)

        np.save(fp_cam + '.npy', cam)

        print("Sample ", filenames_allfiles[i], " true ", y, "pred ", y_pred, i, "/", len(filenames_allfiles))

    print('Grad cam completed')


def run_training():
    """
    Run the cross validation given the data, x_data, y, the shufflesplit object, the
    modified model, number of epochs and whether to use the gpu. Specifically
    the model is trained for the number epochs per cross validation set as well as
    tested. The accuracies and models are saved.
    """
    print("\nRunning with model: {} {}\n".format(args.model_name, args.train_config))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # create the directory of the model type in case it does not exist
    try:
        os.makedirs(args.fp_save)
        print("Creating directory: ", args.fp_save)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #--------------------------------------------------------------------------------------------------------------#
    # Data loading and preprocessing

    filenames_allfiles, y = get_data()

    # # get only filenames from tuesday, as only day with every sample ID
    # filenames_allfiles_tuesday = np.array([name for name in glob.glob(args.fp_data + "2/*")])
    # y_tuesday = []
    # for fname in filenames_allfiles_tuesday:
    #     fname = fname.split(".JPEG")[0].split("/")[-1].replace("_", ",")
    #     y_tuesday.append(get_dai_label(fname))
    # y_tuesday = np.array(y_tuesday)
    # y_tuesday[y_tuesday > 0] = 1
    #
    # # create shufflesplit object, based only on tuesday first
    # rs = StratifiedShuffleSplit(n_splits=args.n_cvruns, test_size=0.25, random_state=args.seed)
    #
    # cv_splits = [[train_index, test_index] for train_index, test_index in rs.split(filenames_allfiles_tuesday, y_tuesday)]
    #
    # # get the desired cv split
    # train_index, test_index = cv_splits[args.cv_run]
    #
    # # now put all filenames of the plant ids previously put into train and test also into overall train and test
    # # i.e. plant Z1_0_0_0 from train_index will be in train_index_all regardless of the day, so 1-5
    # train_index_all = []
    # test_index_all = []
    # for idx in train_index:
    #     fname = filenames_allfiles_tuesday[idx]
    #     plant_id = fname.split('/2_')[-1]
    #     train_index_all += [i for i in range(len(filenames_allfiles)) if plant_id in filenames_allfiles[i]]
    # for idx in test_index:
    #     fname = filenames_allfiles_tuesday[idx]
    #     plant_id = fname.split('/2_')[-1]
    #     test_index_all += [i for i in range(len(filenames_allfiles)) if plant_id in filenames_allfiles[i]]
    #
    # train_index = train_index_all
    # test_index = test_index_all
    #
    # train_samples = [fname.split('/')[-1].split('.')[0] for fname in filenames_allfiles[train_index]]
    # test_samples = [fname.split('/')[-1].split('.')[0] for fname in filenames_allfiles[test_index]]

    # load and permutate data
    # load data and permutate
    if not args.rrr:
        filenames_allfiles, x_data, y, perm_ids = imread_from_fp_rescale_rotate_flatten(
            fplist=filenames_allfiles, y=y,
            rescale_size=224, n_rot_per_img=args.n_rotations
        )
    else:
        print("Loading masks as well ...")
        filenames_allfiles, x_data, masks, y, perm_ids = imread_from_fp_rescale_rotate_flatten_returnmasks(
            fplist=filenames_allfiles, fp_mask=args.fp_mask, y=y, rescale_size=224,
            n_rot_per_img=args.n_rotations, rrr=args.rrr, model=args.model_name
        )

    # get split from corresponding cv run, based on permutated filenames_allfiles
    train_index, test_index, train_samples, test_samples = get_data_split(filenames_allfiles)

    print("Images read")
    print("Size of data:" + str(sys.getsizeof(x_data) * 1e-9) + " GB")

    print("\n----------------------------------------------------------------")
    print("Model: {}".format(args.model_name))
    print("Cross validation round: " + str(args.cv_run))

    print("StandardScaler being fit using training data...")
    std_scaler = preprocessing.StandardScaler()
    std_scaler = std_scaler.fit(x_data[train_index, :])

    print("StandardScaler transform being applied ...")
    x_norm = reshape_flattened_to_tensor_rgb(std_scaler.transform(x_data), width_height=224)

    print("Data reshaped ...")

    #--------------------------------------------------------------------------------------------------------------#
    # Loading model

    # load pretrained model
    print("Loading the pretrained torchvision model.")
    model, _, model_params = load_model(num_classes=args.num_classes, feature_extract=False, use_pretrained=True)
    model.target_output = None

    optimizer = torch.optim.Adam(model_params, lr=args.lr, amsgrad=True, weight_decay=1e-5)

    # set optimizer and criterion
    weights = [np.sum(y[train_index] == i)/len(y[train_index]) for i in np.arange(0, args.num_classes)]
    class_weights = torch.FloatTensor(weights)
    if args.cuda:
        class_weights = class_weights.cuda()

    if args.rrr:
        print("Using NLLLoss")
        criterion_train = nn.NLLLoss(weight=class_weights).cuda()
        criterion_test = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Using CrossEntropyLoss")
        criterion_train = nn.CrossEntropyLoss(weight=class_weights)
        criterion_test = nn.CrossEntropyLoss(weight=class_weights)

    if not args.rrr:
        scheduler = MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1, last_epoch=-1)

    if args.cuda:
        model.cuda()

    print('Model successfully loaded ...')

    #--------------------------------------------------------------------------------------------------------------#
    # Setting up Dataloaders

    # create tensors for dataloaders
    train_tensors = (torch.tensor(x_norm[train_index, :, :, :]), torch.tensor(y[train_index]))
    test_tensors = (torch.tensor(x_norm[test_index, :, :, :]), torch.tensor(y[test_index]))

    print("Tensors created")

    # if required add masks
    if args.rrr:
        masks = np.reshape(masks, (masks.shape[0], 14, 14))
        train_tensors = train_tensors + (torch.tensor(masks[train_index]),)
        test_tensors = test_tensors + (torch.tensor(masks[test_index]),)

    train_dataset = CustomRGBDataset(tensors=train_tensors)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataset = CustomRGBDataset(tensors=test_tensors)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size)

    print('Data successfully loaded ...')

    #--------------------------------------------------------------------------------------------------------------#
    # Training process

    print('Training beginning ...')

    last_checkpoint_fp = 0
    last_best_checkpoint_fp = 0

    last_test_acc = -10

    raloss_array = np.ones(args.n_epochs)
    rrloss_array = np.ones(args.n_epochs)
    bal_acc_train_array = np.ones(args.n_epochs)
    bal_acc_test_array = np.ones(args.n_epochs)

    # train on number of epochs
    for epoch in np.arange(1, args.n_epochs+1):
        with torch.set_grad_enabled(True):
            epoch_train_loss, epoch_train_acc, epoch_ra_loss, epoch_rr_loss = train(model, train_loader,
                                                                                    criterion_train,
                                                                                    class_weights=class_weights,
                                                                                    optimizer=optimizer,
                                                                                    verbose=0)
            print("Epoch: {} Train Loss : {:.4f}  Train Accuracy: {:.4f} "
                  "Train Right Answer Loss: {:.4f} Train Right Reason Loss: {:.4f}".format(epoch, epoch_train_loss,
                                                                                           epoch_train_acc,
                                                                                           epoch_ra_loss,
                                                                                           epoch_rr_loss))
            raloss_array[epoch-1] = epoch_ra_loss
            rrloss_array[epoch - 1] = epoch_rr_loss
            bal_acc_train_array[epoch - 1] = epoch_train_acc

        with torch.set_grad_enabled(False):
            print("Evaluation running...")
            epoch_test_loss, epoch_test_acc = test(model, test_loader, criterion_test, verbose=0)
            print("Epoch: {} Test Loss : {:.4f}  Test Accuracy: {:.4f}".format(epoch, epoch_test_loss, epoch_test_acc))

            bal_acc_test_array[epoch - 1] = epoch_test_acc

            if not args.rrr:
                print('Epoch:', epoch, 'LR:', scheduler.get_lr())
                scheduler.step()

            # save current model
            checkpoint = {'model_state': model.state_dict(),
                          'std_scaler': std_scaler,
                          'optimizer_state': optimizer.state_dict(),
                          'args': args,
                          'epoch': epoch,
                          'test_loss': epoch_test_loss,
                          'test_acc': epoch_test_acc,
                          }

            if epoch > 1:
                # remove checkpoint saved in last epoch
                os.remove(last_checkpoint_fp)

            last_checkpoint_fp = "{}{}_cvnum_{}_epoch_{}_evalbalacc_{}_trainraloss_" \
                                 "{}_trainrrrloss_{}.pth".format(args.fp_save, args.model_name,
                                                                             str(args.cv_run), str(epoch),
                                                                             str(int(1e+4 * round(epoch_test_acc, 4))),
                                                                             str(round(epoch_ra_loss, 4)),
                                                                             str(round(epoch_rr_loss, 4)))


            torch.save(checkpoint, last_checkpoint_fp)

            # if the test acc is higher than any model before store this model seperately
            if epoch_test_acc > last_test_acc and not args.rrr:
                if epoch > 1:
                    # delete last best model
                    os.remove(last_best_checkpoint_fp)

                last_best_checkpoint_fp = "{}{}_cvnum_{}_epoch_{}_evalbalacc_{}_trainraloss_" \
                                          "{}_trainrrrloss_{}_besttestacc.pth".format(args.fp_save, args.model_name,
                                                                     str(args.cv_run), str(epoch),
                                                                     str(int(1e+4 * round(epoch_test_acc, 4))),
                                                                     str(round(epoch_ra_loss, 4)),
                                                                     str(round(epoch_rr_loss, 4)))

                torch.save(checkpoint, last_best_checkpoint_fp)
                last_test_acc = epoch_test_acc
                print("Saving new best model...")

    print("\nTraining " + args.model_name + " finished\n-------------------------------------------\n")


def run_test():
    """
    Evaluate the model indicated in the model checkpoint path
    """
    # load previous model dictionary
    state_dict = torch.load(args.fp_save + args.cp_fname)
    std_scaler = state_dict['std_scaler']

    # load model
    model, _, model_params = load_model(num_classes=args.num_classes, feature_extract=False, use_pretrained=True)
    model.load_state_dict(state_dict['model_state'], strict=True)
    if args.cuda:
        model.cuda()
    print("The model was loaded from checkpoint")

    filenames_allfiles, y = get_data()

    # load and permutate data
    filenames_allfiles, x_data, y, perm_ids = imread_from_fp_rescale_rotate_flatten(
        fplist=filenames_allfiles, y=y,
        rescale_size=224, n_rot_per_img=0
    )

    # get split from corresponding cv run, based on permutated filenames_allfiles
    train_index, test_index, train_samples, test_samples = get_data_split(filenames_allfiles)

    print("StandardScaler being fit using training data...")
    std_scaler = preprocessing.StandardScaler()
    std_scaler = std_scaler.fit(x_data[train_index, :])

    print("StandardScaler transform being applied ...")
    x_norm = reshape_flattened_to_tensor_rgb(std_scaler.transform(x_data), width_height=224)

    #--------------------------------------------------------------------------------------------------------------#
    # Setting up Dataloaders

    # create tensors for dataloaders
    test_tensors = (torch.tensor(x_norm[test_index, :, :, :]), torch.tensor(y[test_index]))

    test_dataset = CustomRGBDataset(tensors=test_tensors)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size)

    print('Data successfully loaded ...')

    #--------------------------------------------------------------------------------------------------------------#
    # compute prediciton on test set

    _, test_acc = test(model, test_loader, criterion_test, verbose=0)

    print(f"Test accuracy of cv run {args.cv_run}: {test_acc:.4f}")


def main():
    create_args()
    if args.train:
        run_training()
    elif args.test:
        run_test()
    if args.gen_cams:
        gen_cams()


if __name__ == "__main__":
    setproctitle.setproctitle('ML CNNs Z dataset')
    main()
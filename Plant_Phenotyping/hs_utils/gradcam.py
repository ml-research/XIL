from PIL import Image
import numpy as np
import torch

from Plant_Phenotyping.hs_utils.misc_functions import save_class_activation_images3D


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        model = self.model
        for module_pos, module in model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        model = self.model
        x = model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):

        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255

        return cam

    def generate_cam_3D(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        # Target for backprop
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(device)
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]  # CDHW e.g. 128,4,13,13
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(2, 3))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for j in range(weights.shape[1]):
            for i, w in enumerate(weights[:, j]):
                cam[j] += w * target[i, j, :, :]

        cam = np.maximum(cam, 0)
        tmp_min_cam = np.min(cam)
        tmp_max_cam = np.max(cam)

        cam = (cam - tmp_min_cam) / (tmp_max_cam - tmp_min_cam)  # Normalize between 0-1
        spectral_cam_median = np.median(cam, axis=(1, 2))
        spectral_cam_mean = np.mean(cam, axis=(1, 2))
        spectral_cam_std = np.std(cam, axis=(1, 2))
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

        cams_resized = list()
        for idx in range(cam.shape[0]):
            cam_tmp = cam[idx]
            cam_resized = Image.fromarray(cam_tmp).resize((input_image.shape[3], input_image.shape[4]), Image.ANTIALIAS)
            cams_resized.append(((np.uint8(cam_resized) / 255) * (tmp_max_cam - tmp_min_cam)) + tmp_min_cam)

        cam = np.array(cams_resized)
        cam = np.repeat(cam, input_image.shape[2]//cam.shape[0], axis=0)

        return cam, input_image.shape[2]//len(cams_resized), (spectral_cam_mean, spectral_cam_std, spectral_cam_median)
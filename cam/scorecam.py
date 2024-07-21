import torch
import torch.nn.functional as F

from cam.properties import ModelInfo
from utils import (
    find_alexnet_layer,
    find_vgg_layer,
    find_resnet_layer,
    find_densenet_layer,
    find_squeezenet_layer,
    find_layer,
    find_googlenet_layer,
    find_mobilenet_layer,
    find_shufflenet_layer
)

class ScoreCAM():

    """
        ScoreCAM

    """

    def __init__(self, model_dict: ModelInfo) -> None:
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        
        self.model_arch = model_dict['architecture']
        self.model_arch.eval()
        if torch.cuda.is_available():
            self.model_arch.cuda()
        self.gradients = dict()
        self.activations = dict()

        def forward_hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
            if torch.cuda.is_available():
                self.activations['value'] = output.cuda()
            else:
                self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            self.target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            self.target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            self.target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            self.target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            self.target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'googlenet' in model_type.lower():
            self.target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif 'shufflenet' in model_type.lower():
            self.target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif 'mobilenet' in model_type.lower():
            self.target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)

        self.target_layer.register_forward_hook(forward_hook)

    def forward(self, input: torch.Tensor, class_idx: int | None=None):
        b, c, h, w = input.size()
        
        # Prediction on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
        else:
            predicted_class = torch.LongTensor([class_idx])
        
        logit = F.softmax(logit, dim=1)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(k):

                # Upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                
                if saliency_map.max() == saliency_map.min():
                    continue
                
                # Normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # How much increase if keeping the highlighted region
                # Prediction on masked input
                output = self.model_arch(input * norm_saliency_map)
                output = F.softmax(output, dim=1)
                score = output[0][predicted_class]

                score_saliency_map += score * saliency_map
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input: torch.Tensor, class_idx: int | None=None):
        return self.forward(input, class_idx)
import torch
import torch.nn.functional as F

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
    def __init__(self, model_name: str, model_architecture: torch.nn.Module, target_layer_name: str) -> None:
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.model_architecture = model_architecture
        self.model_architecture.eval()
        if torch.cuda.is_available():
            self.model_architecture.cuda()
        self.activations = dict()

        def forward_hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
            if torch.cuda.is_available():
                self.activations['value'] = output.cuda()
            else:
                self.activations['value'] = output
            return None

        if 'vgg' in self.model_name.lower():
            self.target_layer = find_vgg_layer(self.model_architecture, self.target_layer_name)
        elif 'resnet' in self.model_name.lower():
            self.target_layer = find_resnet_layer(self.model_architecture, self.target_layer_name)
        elif 'densenet' in self.model_name.lower():
            self.target_layer = find_densenet_layer(self.model_architecture, self.target_layer_name)
        elif 'alexnet' in self.model_name.lower():
            self.target_layer = find_alexnet_layer(self.model_architecture, self.target_layer_name)
        elif 'squeezenet' in self.model_name.lower():
            self.target_layer = find_squeezenet_layer(self.model_architecture, self.target_layer_name)
        elif 'googlenet' in self.model_name.lower():
            self.target_layer = find_googlenet_layer(self.model_architecture, self.target_layer_name)
        elif 'shufflenet' in self.model_name.lower():
            self.target_layer = find_shufflenet_layer(self.model_architecture, self.target_layer_name)
        elif 'mobilenet' in self.model_name.lower():
            self.target_layer = find_mobilenet_layer(self.model_architecture, self.target_layer_name)
        else:
            self.target_layer = find_layer(self.model_architecture, self.target_layer_name)

        self.target_layer.register_forward_hook(forward_hook)

    def forward(self, input: torch.Tensor, class_idx: int | None=None):
        b, c, h, w = input.size()
        
        # Prediction on raw input
        logit = self.model_architecture(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
        else:
            predicted_class = torch.LongTensor([class_idx])
        
        # logit = F.softmax(logit, dim=1)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            # logit = logit.cuda()

        self.model_architecture.zero_grad()
        activations = self.activations['value'] # Get activation maps
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            # Loop through each activation map
            for i in range(k):

                # Upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1) # Get the activation map and restructure to size (b, 1, u, v)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False) # Resize to size (b, 1, h, w)
                
                if saliency_map.max() == saliency_map.min():
                    continue
                
                # Normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # How much increase if keeping the highlighted region
                # Prediction on masked input
                output = self.model_architecture(input * norm_saliency_map)
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
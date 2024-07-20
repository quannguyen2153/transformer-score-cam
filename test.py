from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import *
from cam.scorecam import *

image_dir = Path("images")
sample_image = Path("ILSVRC2012_val_00002193.JPEG")
output_dir = Path("output")
alexnet_output = output_dir / Path("alexnet.png")
vgg_output = output_dir / Path("vgg.png")
resnet_output = output_dir / Path("resnet.png")

# alexnet
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).eval()
alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_10',input_size=(224, 224))
alexnet_scorecam = ScoreCAM(alexnet_model_dict)

input_image = load_image(image_path=image_dir / sample_image)
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = alexnet(input_).max(1)[-1]

scorecam_map = alexnet_scorecam(input_)
basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=alexnet_output)
print(f"ScoreCAM output of AlexNet is saved as {alexnet_output}.")

# vgg
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()
vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29',input_size=(224, 224))
vgg_scorecam = ScoreCAM(vgg_model_dict)

input_image = load_image(image_path=image_dir / sample_image)
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = vgg(input_).max(1)[-1]

scorecam_map = vgg_scorecam(input_)
basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=vgg_output)
print(f"ScoreCAM output of VGG16 is saved as {vgg_output}.")

# resnet
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
resnet_model_dict = dict(type='resnet18', arch=resnet, layer_name='layer4',input_size=(224, 224))
resnet_scorecam = ScoreCAM(resnet_model_dict)

input_image = load_image(image_path=image_dir / sample_image)
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = resnet(input_).max(1)[-1]

scorecam_map = resnet_scorecam(input_)
basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=resnet_output)
print(f"ScoreCAM output of ResNet is saved as {resnet_output}.")
import torch
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
from Helen_transform import  New_Resize, New_ToTensor
import torchvision
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image
from dataset import TwoStepData
from torch.utils.data import DataLoader
from train import ModelTrain
import torch.nn.functional as F
import numpy as np


def imshow(inp, title=None):
    """Imshow ."""
    inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots a


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_file_root = "/data3/vimg18/python_projects/icnn-face/checkpoints_8bd0cf02/"
state_file_1 = os.path.join(state_file_root,
                            "best.pth.tar")
img_root_dir = "/data1/datas"
part_root_dir = "/data1/facial_parts"
txt_file = 'testing.txt'
test_dataset = TwoStepData(txt_file=txt_file,
                           img_root_dir=img_root_dir,
                           part_root_dir=part_root_dir,
                           transform=transforms.Compose([
                               New_Resize((64, 64)),
                               New_ToTensor()
                           ])
                           )
test_dataloader = DataLoader(test_dataset, batch_size=16,
                             shuffle=False, num_workers=4)

test = ModelTrain()
test.load_state(state_file_1)

test_img = next(iter(test_dataloader))
img = test_img['image'].to(device)
orig = test_img['orig'].to(device)
labels = test_img['labels'].to(device)
y = test.model(img, orig)

out2 = torchvision.utils.make_grid(img)
imshow(out2)

theta = test.model.model[0].get_all_theta()
for i in range(8):

    grid = F.affine_grid(theta[:, i], size=(img.shape[0], 3, 64, 64), align_corners=True).to(device)
    sample = F.grid_sample(input=orig, grid=grid, align_corners=True)
    print("theta%d" % i, theta[:, i])
    out_sample = torchvision.utils.make_grid(sample)
    imshow(out_sample)

print("Ground Truth")
for i in range(8):
    out3 = torchvision.utils.make_grid(torch.unsqueeze(labels[:, i], dim=1))
    imshow(out3)
print("Predict")
for i in range(8):
    out = torchvision.utils.make_grid(torch.unsqueeze(y[:, i], dim=1))
    imshow(out)


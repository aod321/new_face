import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model_1 import FaceModel
from model_2 import Stage2FaceModel
class SelectNet(torch.nn.Module):
    def __init__(self):
        super(SelectNet, self).__init__()
        self.theta = None
        self.sample = None
        self.orig = None

    # Spatial transformer network forward function
    def set_orig(self, orig):
        self.orig = orig

    def get_theta(self):
        return self.theta

    def forward(self, x):
        # input Shape(N,2,2)
        #       out = [[sx,sy],
        #             [tx,ty]]
        #       theta = [[sx,0,tx]
        #                [0,sy,ty]
        #               ]
        self.theta = torch.zeros((x.shape[0],2,3)).to(x.device)
        self.theta[:,0,0] = torch.sigmoid(x[:,0,0])
        self.theta[:,0,2] = torch.tanh(x[:,1,0])
        self.theta[:,1,1] = torch.sigmoid(x[:,0,1])
        self.theta[:,1,2] = torch.tanh(x[:,1,1])
        grid = F.affine_grid(self.theta, size=(x.shape[0],3,64,64), align_corners=True).to(x.device)
        self.sample = F.grid_sample(input=self.orig, grid=grid, align_corners=True).to(x.device)
        assert self.sample.shape == (x.shape[0],3,64,64)

        return self.sample


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")

class Stage1Model(torch.nn.Module):
    def __init__(self):
        super(Stage1Model, self).__init__()
        self.stage1_model = FaceModel()
        state = torch.load("/home/yinzi/stage1_6train/stage1.pth.tar", map_location=device)
        self.stage1_model.load_state_dict(state['model'])
        set_parameter_requires_grad(self.stage1_model, feature_extracting=True)

        self.localization = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1), # 8 x 32 x 32
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),     # 8 x 16 x 16
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=3,stride=2,padding=1),   # 8 x 8 x 8
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),     # 8 x 4 x 4
            nn.ReLU(True)
        )

        # Regressor for the affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )

        self.selectnets = nn.ModuleList([SelectNet() for _ in range(8)])

    def set_orig(self, orig):
        for i in range(8):
            self.selectnets[i].set_orig(orig)

    def get_all_theta(self, print_flag=False):
        theta = []
        for i in range(8):
            theta.append(self.selectnets[i].get_theta())
        theta = torch.stack(theta, dim=0)  # Shape(8,N,2,2)
        theta = torch.transpose(theta, 1, 0)  # Shape(N,8,2,2)
        print("Theta shape:", theta.shape)
        if print_flag:
            print(theta)
        return theta

    def forward(self, x):
        # x input Shape(N,3,64,64)
        out = self.stage1_model(x)
        out = self.localization(out[:, 1:9])
        out = out.view(-1, 128)
        out = self.fc_loc(out)
        out = out.view(-1, 8, 2, 2)
        # output x Shape(N,8,2,2)
        temp = []
        for i in range(8):
            temp.append(self.selectnets[i](out[:, i]))  # input Shape(N,2,2)
        out = torch.stack(temp, dim=0)  # output Shape(8,N,3,64,64)
        del temp
        out = torch.transpose(out, 1, 0)  # output Shape(N,8,3,64,64)
        assert out.shape == (x.shape[0], 8, 3, 64, 64)
        return out


class Stage2Feature(torch.nn.Module):
    def __init__(self):
        super(Stage2Feature,self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Stage2Model(torch.nn.Module):
    def __init__(self):
        super(Stage2Model, self).__init__()
        self.stage2_model = nn.ModuleList([Stage2FaceModel() for _ in range(8)])
        for i in range(8):
            self.stage2_model[i].set_label_channels(1)
    def forward(self, x):
        # x Shape(N,8,3,64,64)
        # Stage2
        stage2_output = []
        for i in range(8):
            stage2_output.append(self.stage2_model[i](x[:, i]))  # input Shape(N,3,64,64)
        stage2_output = torch.stack(stage2_output)  # Shape(8,N,1,64,64)
        stage2_output = torch.squeeze(stage2_output, dim=2)  # Shape(8,N,64,64)
        stage2_output = torch.transpose(stage2_output, 1, 0)  # Shape(N,8,64,64)
        assert stage2_output.shape == (x.shape[0], 8, 64, 64)

        return stage2_output


class TwoStageModel(torch.nn.Module):
    def __init__(self):
        super(TwoStageModel, self).__init__()
        self.model = nn.Sequential(
            Stage1Model(),
            Stage2Model()
        )

    def forward(self, x, orig):
        self.model[0].set_orig(orig)
        y = self.model(x)
        return y

from template import TemplateModel
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from model import TwoStageModel
from torchvision import transforms
import torch.optim as optim
import os
import argparse
from dataset import TwoStepData
from Helen_transform import New_Resize, New_ToTensor
from torch.utils.data import DataLoader
import uuid
import numpy as np
torch.backends.cudnn.benchmark=True

uuid = str(uuid.uuid1())[0:8]

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=0, type=int, help="Choose which GPU")
parser.add_argument("--batch_size", default=20, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
parser.add_argument("--momentum", default=0.9, type=int, help="momentum ")
parser.add_argument("--weight_decay", default=0.005, type=int, help="weight_decay ")

args = parser.parse_args()
print(args)
device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

img_root_dir =  "/data1/yinzi/datas"
part_root_dir = "/data1/yinzi/facial_parts"
txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}
twostage_Dataset = {x:TwoStepData(txt_file=txt_file_names[x],
                      img_root_dir=img_root_dir,
                      part_root_dir=part_root_dir,
                      transform=transforms.Compose([
                          New_Resize((64, 64)),
                          New_ToTensor()
                      ])
                     )
                    for x in ['train', 'val']
                   }

twostage_dataloader = {x:DataLoader(twostage_Dataset[x], batch_size=args.batch_size,
                             shuffle=True, num_workers=16)
                       for x in ['train', 'val']
                      }


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):

            self.next_input = {'image': self.next_input['image'].cuda(non_blocking=True).float(),
                               'labels': self.next_input['labels'].cuda(non_blocking=True).float(),
                               'orig': self.next_input['orig'].cuda(non_blocking=True).float()}
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


class ModelTrain(TemplateModel):
    def __init__(self):
        super(ModelTrain, self).__init__()
        self.writer = SummaryWriter('log')
        self.accumulation_steps = 1

        self.model = TwoStageModel().to(device)
        #         self.model = nn.DataParallel(self.model,device_ids)

        # self.optimizer = optim.SGD(self.model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.BCEWithLogitsLoss()

        self.train_loader = twostage_dataloader['train']
        self.eval_loader = twostage_dataloader['val']

        self.device = device

        self.ckpt_dir = "checkpoints_%s" % uuid
        self.display_freq = args.display_freq
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def train_loss(self, batch):
        x, labels = batch['image'], batch['labels']
        orig = batch['orig']
        # Labels Shape(N,8,64,64)
        # x Shape(N,3,64,64)
        assert orig.shape == (x.shape[0], 3, 256, 256)
        pred = self.model(x, orig)
        # pred Shape(N,8,64,64)
        loss = self.criterion(pred,labels)
        # for i in range(8):
        #     loss_list.append(self.criterion(pred[:, i], labels[:, i]))  # Shape(N,64,64)

        return loss, None

    def train(self):
        self.model.train()
        prefetcher = data_prefetcher(self.train_loader)
        self.epoch += 1
        batch = prefetcher.next()
        iteration = 0
        while batch is not None:
            iteration += 1
            self.step += 1
            self.optimizer.zero_grad()
            loss, others = self.train_loss(batch)
            loss.backward()
            self.optimizer.step()
            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_train_%s' % uuid, loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss))
                if self.train_logger:
                    self.train_logger(self.writer, others)
            batch = prefetcher.next()
        torch.cuda.empty_cache()

    def eval_error(self):
        error_list = []
        prefetcher = data_prefetcher(self.eval_loader)
        batch = prefetcher.next()
        test_loss = 0
        iter = 0
        while batch is not None:
            iter += 1
            x, y = batch['image'], batch['labels']
            orig = batch['orig']
            pred = self.model(x, orig)
            # pred Shape(N,8,64,64)
            test_loss += self.metric(pred, y)
            batch = prefetcher.next()
        test_loss /= iter
        return test_loss, None

def start_train():
    train = ModelTrain()
    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')

def resume_train():
    state_file_root = "/data3/yinzi/vimg18/python_projects/icnn-face/checkpoints_53e6a2aa/"
    checkpoint = os.path.join(state_file_root,
                              "best.pth.tar")
    n_epochs = 25
    lr = 0.001
    Next_train = ModelTrain()
    Next_train.optimizer = optim.SGD(Next_train.model.parameters(), lr, momentum=0.9)
    Next_train.scheduler = optim.lr_scheduler.StepLR(Next_train.optimizer, step_size=5, gamma=0.5)
    Next_train.load_state(checkpoint)
    for i in range(n_epochs):
        Next_train.train()
        Next_train.scheduler.step()
        if Next_train.epoch % args.eval_per_epoch == 0:
            Next_train.eval()


if __name__ == '__main__':
    start_train()


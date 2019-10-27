from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataset import TwoStepData
from torch.utils.data import DataLoader
from Helen_transform import New_Resize, New_ToTensor
from model import TwoStageModel

img_root_dir =  "/data1/yinzi/datas"
part_root_dir = "/data1/yinzi/facial_parts"
txt_file_names = {

    'train': "exemplars.txt",
    'val': "tuning.txt"
}

twostage_Dataset = {x: TwoStepData(txt_file=txt_file_names[x],
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
                             shuffle=True, num_workers=10)
                       for x in ['train', 'val']
                      }


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch['image'].to(device), batch['labels'].to(device)
        orig = batch['orig'].to(device)
        optimizer.zero_grad()
        output = model(data, orig)
        loss = F.binary_cross_entropy_with_logits(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    iter = 0
    with torch.no_grad():
        for batch in test_loader:
            iter +=1
            data, target = batch['image'].to(device), batch['labels'].to(device)
            orig = batch['orig'].to(device)
            output = model(data, orig)
            test_loss += F.binary_cross_entropy_with_logits(output, target).item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= iter

    print('\nTest set: Average loss: {:.4f},({:.0f}%)\n'.format(
        test_loss, len(test_loader.dataset)))

global schduer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Train Now')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    twostage_dataloader = {x: DataLoader(twostage_Dataset[x], batch_size=args.batch_size,
                                         shuffle=True, num_workers=10)
                           for x in ['train', 'val']
                           }

    train_loader = twostage_dataloader['train']
    test_loader = twostage_dataloader['val']

    model = TwoStageModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    global schduer
    schduer = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        schduer.step()
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "trained_cnn.pt")


if __name__ == '__main__':
    main()

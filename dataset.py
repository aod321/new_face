from torch.utils.data import Dataset
import jpeg4py as jpeg
import os
import numpy as np
import torchvision
from skimage import io


class TwoStepData(Dataset):

    def __len__(self):
        return len(self.name_list)

    def __init__(self, txt_file, img_root_dir, part_root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        torchvision.set_image_backend('accimage')
        self.name_list = np.loadtxt(os.path.join(img_root_dir, txt_file), dtype="str", delimiter=',')
        self.img_root_dir = img_root_dir
        self.part_root_dir = part_root_dir
        self.transform = transform
        self.label_name = {
            2: 'eyebrow1',
            3: 'eyebrow2',
            4: 'eye1',
            5: 'eye2',
            6: 'nose',
            7: 'mouth',
            8: 'mouth',
            9: 'mouth'
        }

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.img_root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.part_root_dir, '%s' % self.label_name[i],
                                    'labels', img_name,
                                    img_name + "_lbl%.2d.png" % i)
                       for i in range(2, 10)]
        image = jpeg.JPEG(img_path).decode()  # [1, H, W]
        labels = [io.imread(labels_path[i]) for i in range(8)]  # [8, 64, 64]
        sample = {'image': image, 'labels': labels, 'orig': image}

        if self.transform:
            sample = self.transform(sample)
        return sample

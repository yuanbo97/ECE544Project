import matplotlib
# import the necessary packages
from network import Net
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
import os
import json
from PIL import Image
from matplotlib import cm
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pic', required=True)
args = parser.parse_args()

### Set model timestamp here
now = "09_12_14_57" #datetime.now().strftime("%m_%d_%H_%M")

workers = 2
lr = 1e-3
batch_size = 64
num_epochs  = 15

train_split = 0.6
val_split = 0.2
test_split = 0.2
dataPath = "./data"


saved_model_path = './models/' + now + "/epoch_" + str(14) +".pt"

torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    # Set random seed for reproducibility


    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    global filename

    class mydataset(Dataset):
        def __init__(self):
            global filename
            file = os.path.join(dataPath, 'data_preprocessed.json')
            f = open(file)
            self.data = json.load(f)
            f.close()
            self.x = []
            self.y = []
            self.coord = []
            key, val = list(self.data.items())[int(args.pic)]
            print("Testing image file: " + val['filename'])
            filename = val['filename']
            for k2, v2 in val['patches'].items():
                patch_filename = v2['patch_filename']
                img = Image.open(patch_filename)
                self.x.append(transform(img))
                self.y.append(torch.tensor(v2['label'], dtype=torch.float32))
                self.coord.append(torch.tensor((int(k2.split('_')[0]), int(k2.split('_')[1]))))

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.coord[idx]

        def __len__(self):
            return len(self.y)

    test = mydataset()
    test_dataloader = DataLoader(test, batch_size=batch_size,
                            shuffle=False)

    # initialize the Net
    net = Net()
    net.load_state_dict(torch.load(saved_model_path))
    # initialize our optimizer and loss function

    # initialize a dictionary to store training history
    if not os.path.exists("./compare/"):
        os.mkdir("./compare/")
    compareFolder = "./compare/" + now
    isCompareFolderExist = os.path.exists(compareFolder)
    if not isCompareFolderExist:
        os.mkdir(compareFolder)
    # measure how long training is going to take
    startTime = time.time()

    # loop over our epochs

    inf_start = time.time()
    with torch.no_grad():
        # set the model in evaluation mode
        net.eval()
        # initialize a list to store our predictions
        # loop over the test set
        inf_image = np.zeros((1400, 1000, 3))
        GT_image = np.zeros((1400, 1000, 3))
        inf_mask = np.zeros((1400, 1000))
        GT_mask = np.zeros((1400, 1000))
        for (x, y, coord) in test_dataloader:
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            output = net(x)
            output_label = (output.detach().numpy().round())
            for i in range(min(8, len(coord))):
                for j in range(8):
                    if i * 8 + j >= x.shape[0]:
                        break
                    coordx = coord[i * 8 + j][0].numpy()
                    coordy = coord[i * 8 + j][1].numpy()
                    inf_image[coordx:coordx+10, coordy:coordy+10, :] = np.clip(np.transpose(x[i * 8 + j].cpu().detach().numpy(), (1, 2, 0))[11:21, 11:21, :], 0, 1)
                    GT_image[coordx:coordx+10, coordy:coordy+10, :] = inf_image[coordx:coordx+10, coordy:coordy+10, :]
                    inf_label = output_label[i * 8 + j][0].astype(int)
                    GT_label = y[i * 8 + j].cpu().detach().numpy().astype(int)
                    if inf_label == 1:
                        inf_mask[coordx:coordx+10, coordy:coordy+10] = 1
                        for n in range(coordy, coordy+10):
                            inf_image[coordx, n, :] = np.asarray([1., 0., 0.])
                            inf_image[coordx+9, n, :] = np.asarray([1., 0., 0.])
                        for m in range(coordx, coordx+10):
                            inf_image[m, coordy, :] = np.asarray([1., 0., 0.])
                            inf_image[m, coordy+9, :] = np.asarray([1., 0., 0.])
                    if GT_label == 1:
                        GT_mask[coordx:coordx + 10, coordy:coordy + 10] = 1
                        for n in range(coordy, coordy+10):
                            GT_image[coordx, n, :] = np.asarray([1., 0., 0.])
                            GT_image[coordx+9, n, :] = np.asarray([1., 0., 0.])
                        for m in range(coordx, coordx+10):
                            GT_image[m, coordy, :] = np.asarray([1., 0., 0.])
                            GT_image[m, coordy+9, :] = np.asarray([1., 0., 0.])
        inf_end = time.time()

        figure, axis = plt.subplots(1, 2)
        axis[0].imshow(inf_image)
        axis[0].title.set_text('Inference')
        axis[0].axis("off")

        axis[1].imshow(GT_image)
        axis[1].title.set_text('Ground Truth')
        axis[1].axis("off")

        plt.savefig(compareFolder + "/" + filename + "_compare_noBlend.png", dpi=450)

        inf_mask_img = Image.fromarray(np.uint8(cm.gist_earth(inf_mask) * 255))
        inf_image_img = Image.fromarray(np.uint8(inf_image * 255))
        inf_blended = Image.blend(inf_image_img.convert("RGBA"), inf_mask_img.convert("RGBA"), 0.5)

        GT_mask_img = Image.fromarray(np.uint8(cm.gist_earth(GT_mask) * 255))
        GT_image_img = Image.fromarray(np.uint8(GT_image * 255))
        GT_blended = Image.blend(GT_image_img.convert("RGBA"), GT_mask_img.convert("RGBA"), 0.5)

        figure, axis = plt.subplots(1, 2)
        axis[0].imshow(inf_blended)
        axis[0].title.set_text('Inference')
        axis[0].axis("off")

        axis[1].imshow(GT_blended)
        axis[1].title.set_text('Ground Truth')
        axis[1].axis("off")
        plt.savefig(compareFolder + "/" + filename + "_compare_blended.png", dpi=450)

    endTime = time.time()
    totalTime = round(endTime-startTime)
    totalInfTime = round(inf_end - inf_start)
    print("total time consumed to load the model, interpret an image, and display: {} s".format(totalTime))
    print("total inference time: {} s".format(totalInfTime))






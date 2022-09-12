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
from datetime import datetime

now = datetime.now().strftime("%m_%d_%H_%M")
if not os.path.exists("./models"):
    os.mkdir('./models')
os.mkdir('./models/' + now)

workers = 2
lr = 1e-3
batch_size = 64
num_epochs  = 15

train_split = 0.6
val_split = 0.2
test_split = 0.2
dataPath = "./data"
dataroot = "./data/preprocessed/"

torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    class mydataset(Dataset):
        def __init__(self):
            file = os.path.join(dataPath, 'data_preprocessed.json')
            f = open(file)
            self.data = json.load(f)
            f.close()
            self.x = []
            self.y = []
            for key, val in self.data.items():
                print(val['filename'])
                for k2, v2 in val['patches'].items():
                    patch_filename = v2['patch_filename']
                    img = Image.open(patch_filename)
                    self.x.append(transform(img))
                    self.y.append(torch.tensor(v2['label'], dtype=torch.float32))

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

        def __len__(self):
            return len(self.y)


    isExist = os.path.exists("./data/train_dataloader.pt")
    if isExist:
        print("Found data loader backup, loading them...")
        train_dataloader = torch.load("./data/train_dataloader.pt")
        val_dataloader = torch.load("./data/val_dataloader.pt")
        test_dataloader = torch.load("./data/test_dataloader.pt")
    else:
        print("Creating data loader objects...")
        dataset = mydataset()
        numTrain = round(len(dataset) * train_split)
        numVal = round(len(dataset) * val_split)
        numTest = len(dataset) - numTrain - numVal
        train, val, test = random_split(dataset, [numTrain, numVal, numTest], generator=torch.Generator().manual_seed(manualSeed))
        train_dataloader = DataLoader(train, batch_size=batch_size,
                                             shuffle=True)
        val_dataloader = DataLoader(val, batch_size=batch_size,
                                shuffle=False)
        test_dataloader = DataLoader(test, batch_size=batch_size,
                                shuffle=False)

        torch.save(train_dataloader, "./data/train_dataloader.pt")
        torch.save(val_dataloader, "./data/val_dataloader.pt")
        torch.save(test_dataloader, "./data/test_dataloader.pt")

    trainSteps = len(train_dataloader.dataset) // batch_size
    valSteps = len(val_dataloader.dataset) // batch_size

    real_batch = next(iter(train_dataloader))
    '''
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    '''
    figure, axis = plt.subplots(8, 8, constrained_layout = True)

    if not os.path.exists("./testResult"):
        os.mkdir("./testResult")
    isResultFolderExist = os.path.exists("./testResult/" + now)
    if not isResultFolderExist:
        os.mkdir("./testResult/" + now)

    for i in range(8):
        for j in range(8):
            axis[i, j].imshow(np.clip(np.transpose(real_batch[0][i*8+j].numpy(), (1,2,0)), 0, 1))
            axis[i, j].set_xticks([])
            axis[i, j].set_yticks([])
            axis[i, j].set_xlabel("GT: " + str(real_batch[1][i*8+j].numpy().astype(int)))
    plt.savefig("./testResult/" + now + "/demo.png")
    print("demo figured saved")
    print("dataset size:", len(train_dataloader.dataset))

    # initialize the Net
    net = Net()
    print(net)
    # initialize our optimizer and loss function
    optimizer = Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()

    # loop over our epochs
    for e in range(0, num_epochs):
        # set the model in training mode
        net.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for i, data in enumerate(train_dataloader, 0):
            # send the input to the device
            x = data[0].to(device)
            y = data[1].to(device)
            # perform a forward pass and calculate the training loss
            b_size = x.size(0)
            output = net(x).view(-1)
            loss = loss_fn(output, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (output.detach().numpy().round() == y.detach().numpy()).sum().item()



        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            net.eval()
            # loop over the validation set
            for (x, y) in val_dataloader:
                (x, y) = (x.to(device), y.to(device))
                # send the input to the device
                # make the predictions and calculate the validation loss
                output = net(x).view(-1)
                totalValLoss += loss_fn(output, y)
                # calculate the number of correct predictions
                valCorrect +=  (output.detach().numpy().round() == y.detach().numpy()).sum().item()
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_dataloader.dataset)
        valCorrect = valCorrect / len(val_dataloader.dataset)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

        # finish measuring how long training took
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))
        # we can now evaluate the network on the test set
        print("[INFO] evaluating network...")
        # turn off autograd for testing evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            net.eval()

            # initialize a list to store our predictions
            # loop over the test set
            batch = 0
            figure, axis = plt.subplots(8, 8, constrained_layout=True)
            for (x, y) in test_dataloader:
                batch += 1
                if batch > 20:
                    break
                # send the input to the device
                x = x.to(device)
                # make the predictions and add them to the list
                output = net(x)
                output_label = (output.detach().numpy().round())


                for i in range(8):
                    for j in range(8):
                        axis[i, j].imshow(np.clip(np.transpose(x[i * 8 + j].cpu().detach().numpy(), (1, 2, 0)), 0, 1))
                        axis[i, j].set_xticks([])
                        axis[i, j].set_yticks([])
                        axis[i, j].set_xlabel("GT: " + str(y[i * 8 + j].cpu().detach().numpy().astype(int)) + " O: "
                                              + str(output_label[i * 8 + j][0].astype(int)))
                plt.savefig("./testResult/" + now + "/test_epoch_" + str(e) + "_batch_" + str(batch) + ".png")
                plt.cla()
            plt.close('all')
            print("test results for epoch " + str(e) + " are saved")
        saved_model_path = './models/' + now + "/epoch_" + str(e) +".pt"
        torch.save(net.state_dict(), saved_model_path)
        print("model for epoch " + str(e) + " is saved")




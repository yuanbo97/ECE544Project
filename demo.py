import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random
from PIL import Image
from matplotlib import cm
from IPython.display import display
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
print(torch.cuda.is_available())
dataPath = "./data"


class MyData(Dataset):
    def __init__(self):
        file = os.path.join(dataPath, 'data.json')
        f = open(file)
        self.data = json.load(f)
        f.close()
        resh = 700
        resw = 500
        self.imgs = []
        self.gts = []
        self.nums = []
        self.imgsPIL = []
        self.imgsRaw = []
        for key,val in self.data.items():
            try:
                imgPIL = Image.open(os.path.join(dataPath, val['filename']))
                origwidth = imgPIL.size[0]
                origheight = imgPIL.size[1]
                imgPIL = imgPIL.resize((resw,resh),Image.LANCZOS)
                image = torch.Tensor(np.asarray(imgPIL)/255)
                gt = torch.zeros((image.shape[0:2]))
                for k2,v2 in val['regions'].items():
                    rectdata = v2['shape_attributes']
                    x1 = int(np.floor(resw*float(rectdata['x'])/origwidth))
                    y1 = int(np.floor(resh*float(rectdata['y'])/origheight))
                    x2 = int(np.ceil(resw*float(rectdata['x']+rectdata['width'])/origwidth))
                    y2 = int(np.ceil(resh*float(rectdata['y']+rectdata['height'])/origheight))
                    gt[y1:y2,x1:x2] = 1
                print('File: {}; number: {}'.format(val['filename'],len(val['regions'])))
                gtimg = Image.fromarray(np.uint8(cm.gist_earth(gt.numpy())*255))
                self.imgsRaw.append(imgPIL)
                self.imgsPIL.append(Image.blend(imgPIL.convert("RGBA"),gtimg.convert("RGBA"),0.5))
                self.imgs.append(image.permute(2,0,1))
                self.gts.append(gt)
                self.nums.append(len(val['regions']))
            except IOError:
                print('File not found: {}'.format(val['filename']))
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,idx):
        d,g = self.imgs[idx], self.gts[idx].unsqueeze(0)
        return d, g, idx
data = MyData()

plt.figure()
plt.imshow(data.imgsRaw[0])

plt.figure()
plt.imshow(data.imgsPIL[0])
plt.show()

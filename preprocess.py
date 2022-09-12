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

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
print(torch.cuda.is_available())
dataPath = "./data"
if not os.path.exists(dataPath + "/preprocessed"):
    os.mkdir(dataPath + "/preprocessed")

class MyData(Dataset):
    def __init__(self):
        isExist = os.path.exists(dataPath + "/preprocessed/images")
        if isExist:
            print("preprocessing already finished")
            return
        os.mkdir(dataPath + "/preprocessed/images")
        file = os.path.join(dataPath, 'data.json')
        f = open(file)
        self.data = json.load(f)
        f.close()
        resh = 1400
        resw = 1000
        patch_size = 32
        patch_step = 10
        threshold = 50
        left = patch_size // 2 - patch_step // 2
        right = patch_size // 2 + patch_step // 2
        self.imgs = []
        self.gts = []
        self.nums = []
        self.imgsPIL = []
        self.imgsRaw = []
        self.preprocessed = []
        self.ylableJSON = {}
        channels = 3
        ylableJSON_name = os.path.join(dataPath, "data_preprocessed.json")

        for key,val in self.data.items():

            try:
                imgPIL = Image.open(os.path.join(dataPath, val['filename']))
                filename = val['filename']
                print("preprocessing: " + filename)
                self.ylableJSON[filename] = {}
                self.ylableJSON[filename]['filename'] = filename

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
                self.ylableJSON[filename]['patches'] = {}
                for x in range(0, resh, patch_step):

                    for y in range(0, resw, patch_step):
                        coordinate = str(x) + "_" + str(y)
                        cropped_image = torch.ones((patch_size, patch_size, channels))
                        x_start = 0 if x > left else left - x
                        x_end = patch_size if x < resh - right else left + resh - x
                        y_start = 0 if y > left else left - y
                        y_end = patch_size if y < resw - right else left + resw - y
                        xadj = -left + x_start
                        yadj = -left + y_start
                        cropped_image[x_start:x_end, y_start:y_end, :] = image[x+xadj:x + (x_end - x_start)+xadj, y+yadj:y + (y_end - y_start)+yadj, :]
                        self.ylableJSON[filename]['patches'][coordinate] = {}
                        patch_filename = os.path.join(dataPath, 'preprocessed', 'images' , filename.split('.')[0] + "_" + str(x) + "_" + str(y) + ".jpg")
                        self.ylableJSON[filename]['patches'][coordinate]['patch_filename'] = patch_filename
                        PIL_cropped_image = Image.fromarray(np.uint8((cropped_image.numpy())*255))
                        PIL_cropped_image.save(patch_filename)
                        if torch.sum(gt[x:x+patch_step, y:y+patch_step]) > threshold:
                            self.ylableJSON[filename]['patches'][coordinate]['label'] = 1
                        else:
                            self.ylableJSON[filename]['patches'][coordinate]['label'] = 0

            except IOError:
                print('File not found: {}'.format(val['filename']))
            with open(ylableJSON_name, 'w') as fp:
                json.dump(self.ylableJSON, fp)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,idx):
        d,g = self.imgs[idx], self.gts[idx].unsqueeze(0)
        return d, g, idx
data = MyData()


#plt.show()

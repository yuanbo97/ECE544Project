
from torch import nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU


batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 100

# Number of channels in the training images. For color images this is 3
nc = 3

class Net(nn.Module):
	def __init__(self):
		# call the parent constructor
		super(Net, self).__init__()
		self.main = nn.Sequential(
			Conv2d(in_channels=nc, out_channels=20, kernel_size=(5, 5)),
			ReLU(),
			MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
			Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
			ReLU(),
			MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
			nn.Flatten(),
			Linear(in_features=1250, out_features=32),
			ReLU(),
			Linear(in_features=32, out_features=1),
			nn.Sigmoid(),
		)
	def forward(self, x):
		return self.main(x)
"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer to match the dimensions
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    
class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        device = torch.device("cuda:0")
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        def conv_lay_relu(inp,out,kernel,stride,pad):
            return nn.Sequential(
            nn.Conv2d(
                in_channels=inp,
                out_channels=out,
                kernel_size=kernel,
                stride=stride,
                padding=pad
                ),
            nn.BatchNorm2d(out),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #ResidualBlock(out,out)
            )
        
        def conv_lay_relu_pool(inp,out,kernel,stride,pad,ker_max,stri_max):
            return nn.Sequential(
            nn.Conv2d(
                in_channels=inp,
                out_channels=out,
                kernel_size=kernel,
                stride=stride,
                padding=pad
                ),
            nn.BatchNorm2d(out),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            ResidualBlock(out,out),
            nn.MaxPool2d(ker_max,stri_max)
            )

        def up_samp_relu(inp,out,kernel,stride,pad):
            return nn.Sequential(
                nn.ConvTranspose2d(
                inp,
                out,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                output_padding=0
                ),
                nn.BatchNorm2d(out),
                #nn.Dropout(0.5),
                nn.ReLU(inplace=True),
            )

        def conv_lay_relu_UP(inp,out,kernel,stride,pad):
            return nn.Sequential(
            nn.Conv2d(
                in_channels=inp,
                out_channels=out,
                kernel_size=kernel,
                stride=stride,
                padding=pad
                ),
            nn.BatchNorm2d(out),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #ResidualBlock(out,out),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )


        device = torch.device("cuda:0")
        self.encoder=models.resnet18(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

         # Remove fully connected layer and global average pooling
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-3])

        layers=[]
        layers.append(up_samp_relu(256,128,4,2,self.hp["padding"]))
        layers.append(up_samp_relu(128,64,4,2,self.hp["padding"]))
        layers.append(up_samp_relu(64,32,4,2,self.hp["padding"]))
        layers.append(nn.ConvTranspose2d(32,num_classes,4,2,1))
        self.decoder=nn.Sequential(*layers)
        self.decoder.to(device)

        """"
        device = torch.device("cuda:0")
        layers=[]
        layers.append(conv_lay_relu(3,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"],self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu(self.hp["out_channels"],self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"]*2,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*2,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"]*4,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))

        self.encoder=nn.Sequential(*layers)
        self.encoder.to(device)

        layers=[]
        layers.append(conv_lay_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"]*2,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*2,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*2,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"],self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"],num_classes,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        self.decoder=nn.Sequential(*layers)
        self.decoder.to(device)
        
        ----------

        layers=[]
        layers.append(conv_lay_relu(3,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"],self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*2,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"]*4,self.hp["out_channels"]*8,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        self.encoder=nn.Sequential(*layers)

        layers=[]
        layers.append(conv_lay_relu(self.hp["out_channels"]*8,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"]*2,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*2,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"],num_classes,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        self.decoder=nn.Sequential(*layers)
        ----
        layers=[]
        layers.append(conv_lay_relu_pool(3,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"],self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"]*2,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"]*4,self.hp["out_channels"]*8,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        self.encoder=nn.Sequential(*layers)

        layers=[]
        layers.append(up_samp_relu(self.hp["out_channels"]*8,self.hp["out_channels"]*8,self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*8,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*4,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"]*2,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"]*2,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(up_samp_relu(self.hp["out_channels"],self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride_up"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"],self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"],num_classes,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        self.decoder=nn.Sequential(*layers)
        
        layers=[]
        layers.append(conv_lay_relu_pool(3,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"],self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"]*2,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        layers.append(conv_lay_relu_pool(self.hp["out_channels"]*4,self.hp["out_channels"]*8,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"],self.hp["kernel_max"],self.hp["stride_max"]))
        self.encoder=nn.Sequential(*layers)
        self.encoder=nn.Sequential(*layers)
        self.encoder.to(device)

        layers=[]
        layers.append(conv_lay_relu_UP(self.hp["out_channels"]*8,self.hp["out_channels"]*8,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))

        layers.append(conv_lay_relu_UP(self.hp["out_channels"]*8,self.hp["out_channels"]*4,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu_UP(self.hp["out_channels"]*4,self.hp["out_channels"]*2,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu_UP(self.hp["out_channels"]*2,self.hp["out_channels"],self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        layers.append(conv_lay_relu(self.hp["out_channels"],num_classes,self.hp["kernel_size"],self.hp["stride"],self.hp["padding"]))
        self.decoder=nn.Sequential(*layers)
        self.decoder.to(device)
        """
     

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x=self.encoder(x)
        x=self.decoder(x)
        
        

        return x

    # @property
    def is_cuda(self):
         """
         Check if model parameters are allocated on the GPU.
         """
         return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    #from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")

    
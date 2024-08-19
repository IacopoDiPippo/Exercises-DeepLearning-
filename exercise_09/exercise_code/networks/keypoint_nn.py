"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import torch.nn.init as init

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - (kernel_size - 1) - 1 + 2 * padding) // stride + 1


        def conv_sandwich(inp, out, kernel_size,stride,pad):
            conv=nn.Conv2d(inp,out,kernel_size,stride,pad)
            nn.init.kaiming_normal_(conv.weight,nonlinearity="relu")
            return nn.Sequential(
                conv,
                nn.MaxPool2d(2,2),
                nn.ReLU()
            )

        layers=[]
        layers.append(conv_sandwich(1,32,3,1,1))
        layers.append(conv_sandwich(32,64,3,1,1))
        layers.append(conv_sandwich(64,128,3,1,1))
        layers.append(conv_sandwich(128,256,3,1,1))
        self.convs=nn.Sequential(*layers)
        self.fc1=nn.Sequential(nn.Linear(256*6*6,256),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(256,30))

        nn.init.kaiming_normal_(self.fc1[0].weight,nonlinearity="relu")
        nn.init.xavier_normal_(self.fc2[0].weight)
        #

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        
        x=self.convs(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)


        return x
    
    

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        self.optimizer=torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["decay_rate"]
            )

           


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)


def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - (kernel_size - 1) - 1 + 2 * padding) // stride + 1





def cazzo(self,out):

    out_size = 96
    if True==True:
        out_size = conv2d_size_out(out_size, self.hparams["kernel_size"], self.hparams["stride"], padding=1) // 2
        out_size = conv2d_size_out(out_size, self.hparams["kernel_size"] - 1, self.hparams["stride"], padding=1) // 2
        out_size = conv2d_size_out(out_size, self.hparams["kernel_size"] - 2, self.hparams["stride"], padding=1) // 2
        out_size = conv2d_size_out(out_size, self.hparams["kernel_size"] - 3, self.hparams["stride"], padding=1) // 2
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.hparams["out_channels"],
                kernel_size=self.hparams["kernel_size"],
                stride=self.hparams["stride"],
                padding=2
            ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3),
            #nn.Dropout(hparams["drop_out"]),
            nn.Conv2d(
                in_channels=self.hparams["out_channels"],
                out_channels=self.hparams["out_channels"] * 2,
                kernel_size=self.hparams["kernel_size"],
                stride=self.hparams["stride"],
                padding=2                
            ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Dropout(hparams["drop_out"]*2),
            nn.Conv2d(
                in_channels=self.hparams["out_channels"]*2,
                out_channels=self.hparams["out_channels"] * 2,
                kernel_size=self.hparams["kernel_size"],
                stride=self.hparams["stride"],
                padding=1                 
            ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(hparams["drop_out"]*3),
            nn.Conv2d(
                in_channels=self.hparams["out_channels"] * 2,
                out_channels=self.hparams["out_channels"] * 4,
                kernel_size=self.hparams["kernel_size"]-1,
                stride=self.hparams["stride"],
                padding=1                 
            ),
            nn.PReLU(), 
            nn.Flatten(),   
            nn.Linear(10368,256),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(256,30)
        )    
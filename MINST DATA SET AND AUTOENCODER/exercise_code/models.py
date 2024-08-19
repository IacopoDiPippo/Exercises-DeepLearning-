import torch
import pytorch_lightning as pl

import torch.nn as nn
import numpy as np
import torch.optim as optim
def weights_init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')  # He initialization with uniform distribution
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Inizializzazione del bias a zero
class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        # Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 



        self.encoder=nn.Sequential(
            nn.Linear(self.input_size,hparams['n_hidden']),
            nn.BatchNorm1d(hparams["n_hidden"]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hparams['n_hidden'], hparams["n_hidden2"]),
            nn.BatchNorm1d(hparams["n_hidden2"]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hparams["n_hidden2"],latent_dim)
        )



    # Funzione per l'inizializzazione dei pesi usando He Initialization
    def weights_init_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')  # He initialization with uniform distribution
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Inizializzazione del bias a zero
                
    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # Initialize your decoder!                                       #
        ########################################################################


        self.decoder=nn.Sequential(
            nn.Linear(latent_dim,hparams['n_hidden2']),
            nn.BatchNorm1d(hparams["n_hidden2"]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hparams['n_hidden2'], hparams["n_hidden"]),
            nn.BatchNorm1d(hparams["n_hidden"]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hparams["n_hidden"],output_size)
        )

        

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        #  Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        reconstruction=self.encoder(x)
        reconstruction=self.decoder(reconstruction)

      
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        #  Define your optimizer.                                         #
        ########################################################################
        self.optimizer=optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["decay_rate"]
            )

       

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        #                                                                 #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #
        ae_trainer=pl.Trainer(
            max_epochs=30,
            gpus=1 if torch.cuda.is_available() else None,
            logger=ae_logger
        )


 
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        #                                                               #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
                                                         #
        ########################################################################


        self.eval()

        with torch.no_grad():
        
            images = batch # Get the images and labels from the batch, in the fashion we defined in the dataset and dataloader.
            images= images.to(self.device) # Send the data to the device (GPU or CPU) - it has to be the same device as the model.

        # Flatten the images to a vector. This is done because the classifier expects a vector as input.
        # Could also be done by reshaping the images in the dataset.
            images = images.view(images.shape[0], -1) 

            pred=self.forward(images)
            loss=loss_func(pred,images)

    
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        ########################################################################
        #                                                               #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################

        self.model = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, hparams["n_hidden"]),
            nn.BatchNorm1d(hparams["n_hidden"]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hparams["n_hidden"], hparams["n_hidden1"]),
            nn.BatchNorm1d(hparams["n_hidden1"]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hparams["n_hidden1"], 10)
            )

        #self.model.apply(weights_init_he)
      

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    # Funzione per l'inizializzazione dei pesi usando He Initialization
    def weights_init_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')  # He initialization with uniform distribution
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Inizializzazione del bias a zero


    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        self.optimizer=optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["decay_rate"]
            )


    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

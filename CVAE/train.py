''' This code contains the implementation of conditional CVAE
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F

from models import CVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 40         
N_EPOCHS = 200          
INPUT_DIM = 1214        
HIDDEN_DIM_1 = 512    
HIDDEN_DIM_2 = 512      
LATENT_DIM = 40         
COND_DIM = 40          
lr = 1e-4               
weight_decay = 1e-3 

def load_data(path):
    dir = [f.path for f in os.scandir(path) if f.is_dir()]
    
    chainA_list = []
    chainB_list = []
    cond_list = []
    
    for folder in dir:
        chainA = pd.read_pickle(folder + "/chainA.pkl")
        chainB = pd.read_pickle(folder + "/chainB.pkl")
        cond = pd.read_pickle(folder + "/cond.pkl")
        
        chainA_list.append(chainA)
        chainB_list.append(chainB)
        cond_list.append(cond)
        
        
    return chainA_list, chainB_list, cond_list



def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    criterion = nn.MSELoss()

    RCL_chainA = criterion(reconstructed_x[:-54], x[:-54])
    RCL_chainB = criterion(reconstructed_x[-54:], x[-54:])  
    
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    al = 1
    be = 100

    return al*RCL_chainA + be*RCL_chainB + KLD

model = CVAE(encoder_layer_sizes=[INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2],
        latent_size=LATENT_DIM,
        decoder_layer_sizes=[HIDDEN_DIM_2, HIDDEN_DIM_1, INPUT_DIM],
        cond_size=COND_DIM).to(device)
        
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



def train():
   
    path = '...'  
    
    # load data
    chainA_list, chainB_list, cond_list = load_data(path)
    
    # loss of the epoch
    train_loss = 0

    for chainA,chainB,cond in zip(chainA_list, chainB_list, cond_list):
        
        # Make sure the size of condition var is correct.
        if(cond.shape[1] == COND_DIM):
            # update the gradients to zero
            optimizer.zero_grad()        
            
            for i in range(chainA.shape[0]):
                   
                # Get input and convert it to float
                a = torch.tensor(chainA.iloc[i,:].values).float()
                b = torch.tensor(chainB.iloc[i,:].values).float()
                x = torch.cat((a, b), dim=-1)
                x = x.to(device)
                
                # Get condition and convert it to float
                y = torch.tensor(cond.iloc[0,:].values).float()
                y = y.to(device)

                # forward pass
                reconstructed_x, z_mu, z_var, z = model(x, y)
                            
                # loss
                loss = calculate_loss(x, reconstructed_x, z_mu, z_var)

                # backward pass
                loss.backward()
            
            # update the weights
            optimizer.step()
            train_loss += loss.item()

    return train_loss

for e in range(N_EPOCHS):
    train_loss = train()
    print(f'Epoch {e}, Train Loss: {train_loss:.2f}')

full_model_file = '...'

torch.save(model,full_model_file)

import torch
import torch.nn as nn
from torch.autograd import Variable

X = 41
Y = 41
Z = 6
lamda=1e-3
epoch_num = 100
learing_rate = 0.1

# Model definition

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
			nn.Linear(X*Y*Z, 5043),nn.PReLU(),
			nn.Linear(5043,3125),nn.PReLU(),
			nn.Linear(3125,1600),nn.PReLU(),
			nn.Linear(1600,800),nn.PReLU(),
			nn.Linear(800,400),nn.PReLU(),
			nn.Linear(400,200),nn.PReLU(),
			nn.Linear(200,100),nn.PReLU(),
			nn.Linear(100,50))
			
        self.decoder = nn.Sequential(
			nn.Linear(50,100),nn.PReLU(), 
			nn.Linear(100,200),nn.PReLU(),
			nn.Linear(200,400),nn.PReLU(),
			nn.Linear(400,800),nn.PReLU(),
			nn.Linear(800,1600),nn.PReLU(),
			nn.Linear(1600,3125),nn.PReLU(),
			nn.Linear(3125,5043),nn.PReLU(),
			nn.Linear(5043,X*Y*Z))	

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded	
		
#Loss function for CAE	
	
def loss_function(w,x,y):
	loss_func = torch.nn.MSELoss()
	mse = loss_func(y, x)
	cont_loss = torch.sum(Variable(w)**2, dim=1).sum().mul_(lamda)
	return mse + cont_loss

# Training using single workspace
# data: workspace's representation

def trainAE(data):		
	autoencoder = AutoEncoder()

	optimizer = torch.optim.Adagrad(autoencoder.parameters(), lr= learing_rate) 

	iter = 1
	
	for epoch in range(epoch_num):
		optimizer.zero_grad()               
		train = torch.as_tensor(data.ravel(),dtype=torch.float)
		encoded, decoded = autoencoder(train)
		w = autoencoder.state_dict()['encoder.14.weight']
		loss = loss_function(w,decoded, train)
		loss = loss_func(decoded, train)
		print('epoch: ', epoch, 'iter:,' , iter, 'loss: ', loss.item())
		loss.backward()                    
		optimizer.step()                    
		iter = iter+1
	return autoencoder

# Training using multiple workspace and save the model during the process	
# data: list of workspace's representations
# file_part: uncompleted training model
# file_full: completed training model

def trainAEList(data,file_part):		
	autoencoder = AutoEncoder()

	optimizer = torch.optim.Adagrad(autoencoder.parameters(), lr= learing_rate)

	iter = 1
	
	for epoch in range(epoch_num):
		for i in data:
			optimizer.zero_grad()               
			train = torch.as_tensor(i.ravel(),dtype=torch.float)
			encoded, decoded = autoencoder(train)
			w = autoencoder.state_dict()['encoder.14.weight']
			loss = loss_function(w,decoded, train)
			print('epoch: ', epoch, 'iter:' , iter, 'loss: ', loss.item())
			loss.backward()                    
			optimizer.step()                    
			iter = iter+1
		if (epoch % 10) == 0:
			torch.save({'epoch': epoch,'model_state_dict': autoencoder.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss},file_part)
	
	return autoencoder

# Continuing training using multiple workspace with provided model and save the model during the process	
# state: state dictionary of the input model
# data: list of workspace's representation
# file_part: uncompleted training model
# file_full: completed training model	

def trainContAEList(state,data,file_part,file_full):		
	autoencoder = AutoEncoder()

	optimizer = torch.optim.Adagrad(autoencoder.parameters(), lr= learing_rate)

	iter = 1
	
	autoencoder.load_state_dict(state['model_state_dict'])
	optimizer.load_state_dict(state['optimizer_state_dict'])
	epoch_last = state['epoch']
	loss = state['loss']
	
	autoencoder.train()
	
	for epoch in range(epoch_last+1,epoch_num):
		for i in data:
			optimizer.zero_grad()               
			train = torch.as_tensor(i.ravel(),dtype=torch.float)
			encoded, decoded = autoencoder(train)

			w = autoencoder.state_dict()['encoder.14.weight']
			loss = loss_function(w,decoded, train)

			print('epoch: ', epoch, 'iter:' , iter, 'loss: ', loss.item())
			loss.backward()                    
			optimizer.step()                    
			iter = iter+1
		if (epoch % 10) == 0:
			torch.save({'epoch': epoch,'model_state_dict': autoencoder.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss},file_part)
	
	torch.save({'model_state_dict': autoencoder.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss},file_full)

	return autoencoder	
	
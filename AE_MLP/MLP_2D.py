import torch
import torch.nn as nn

input = 14; # Length of samples and its encoded workspace, this is for 2DS
output = 2; # Valid or invalid
epoch_num = 100
learning_rate = 0.1

# Model definition
 
class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		
		self.fc = nn.Sequential(
			nn.Linear(input_size,6),nn.PReLU(),nn.Dropout(),
			nn.Linear(6,4),nn.PReLU(),nn.Dropout(),
			nn.Linear(4,output_size))
		self.softmax = nn.Softmax(dim=0)
        
	def forward(self, x):
		out = self.fc(x)
		return out

# Training using samples with their encoded workspace, and their label, and save the model during the process	
# data: list of samples and their encoded workspace
# predict: correspond samples' label
# file_part: uncompleted training model
# file_full: completed training model	
		
def trainMLP(data,predict,file_part,file_full):

	model = MLP(input,output)
	
	criterion = torch.nn.CrossEntropyLoss()
	
	optimizer = torch.optim.Adagrad(model.parameters(), lr= learning_rate)
	
	iter = 1
	
	for epoch in range(epoch_num):
		for i in range(0,len(data)):
			for j in range(0,len(data)):
				i_torch = torch.as_tensor(data[i][j],dtype=torch.float)
				predict_torch = torch.as_tensor(predict[i][j],dtype=torch.float)
				optimizer.zero_grad()
				y_pred = model(i_torch)  
				loss = criterion(y_pred, predict_torch)
				print('epoch: ', epoch, 'iter:' , iter, 'loss: ', loss.item())
				loss.backward()
				optimizer.step()
				iter = iter + 1
		if (epoch % 10) == 0:
			torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss},file_part)
			
	torch.save({'model_state_dict': autoencoder.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss},file_full)
	return model


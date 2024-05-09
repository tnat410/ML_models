import numpy as np
from MLP_3D import trainMLP
from CAE_3D import *
import os
import torch
from ultility import *

sample_valid_array_path = ".\\train_Valid\\"
sample_invalid_array_path = ".\\train_Invalid\\"

train_encode_file = 'train_encode.npy'
test_encode_file = 'test_encode.npy'

all_sample_file = 'train_sample.npy'
all_output_file = 'train_label.npy'

file = sample_valid_array_path+"train.txt"

train_model_file = 'MLP_train_model_new.pth'
full_model_file = 'MLP_full_model_new.pth'

model_path = 'MLP_full_model_new.pth'

file_name = []


with open(file, 'r') as f:
	for line in f:
		name = line.rstrip()
		file_name.append(name)
	
	
all_sample = []
all_output = []
encode = np.load(train_encode_file)
all_sample_array = np.load(all_sample_file)
all_output_array = np.load(all_output_file)

def getInput():
	j = 0
	for i in file_name:		
		input_valid = sample_valid_array_path+i+'_valid.0.map'
		input_invalid = sample_invalid_array_path+i+'_invalid.0.map'
		in_val,num_val = file_to_array(input_valid)
		in_inval,num_inval = file_to_array(input_invalid)
		input = np.append(in_val,in_inval,axis = 0)
		obs = np.ones((200,50))*encode[j]
		input = np.column_stack((input,obs))
		all_sample.append(input)
		
		l_1 = [0.0,1.0]*num_val #Label 1 for valid samples
		l_0 = [1.0,0.0]*num_inval #Label 0 for invalid samples
		output = np.append(l_1,l_0,axis = 0)
		#Reshape the output in to 2d array
		output = np.reshape(output, (num_val+num_inval, 2))
		all_output.append(output)
		j = j + 1
		
		all_sample_array = np.asarray(all_sample)
		#np.save(all_sample_file,all_sample_array)

		all_output_array = np.asarray(all_output)
		#np.save(all_output_file,all_output_array)	
	return all_sample_array, all_output_array

all_sample_array, all_output_array = getInput()		
model = trainMLP(all_sample_array,all_output_array,train_model_file,full_model_file)
torch.save(model,full_model_file)





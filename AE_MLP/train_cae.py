from CAE_ import trainAEList
import numpy as np
import os
import torch
from ultility import *

env_train_array_path = ".\\Env_array_train\\"
env_test_array_path = ".\\Env_array_test\\"

train_model_file = 'train_model.pth'
full_model_file = 'full_model.pth'


train_encode_file = '.\\train_encode.npy'
test_encode_file = '.\\test_encode.npy'

file_train = env_train_array_path+"train.txt"
file_test = env_test_array_path+"test.txt"

file_train_name = []
file_test_name = []


with open(file_train, 'r') as f:
	for line in f:
		name = line.rstrip()
		file_train_name.append(name)

train = []
for i in range(0,30):
	array_file = env_train_array_path+file_train_name[i]+".npy"
	t = np.load(array_file)
	train.append(t)
	
	
model = trainAEList(train,train_model_file)
torch.save(model,full_model_file)





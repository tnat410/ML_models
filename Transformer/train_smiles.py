from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.optim as optim
from RobertaSmile import *

class MyDataset(Dataset):

    def __init__(self, data, tokenizer):
        text, labels = data
        self.examples = tokenizer(text=text,text_pair=None,truncation=True,padding="max_length",max_length=256,return_tensors="pt")
        self.smiles = text
        self.labels = torch.tensor(labels, dtype=torch.long)


    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.smiles[index], self.labels[index]

# Prepare a batch from train datasets
def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1) for key, value in batch[0].items()}
    inputs["smiles"] = batch[1]
    inputs["labels"] = batch[2]

    return inputs

data_df = pd.read_csv("...", header=0)
train_samples = (data_df.iloc[:, 0].astype(str).tolist(), data_df.iloc[:, 1].tolist())
train_dataset = MyDataset(train_samples,tokenizer)

tokenizer = AutoTokenizer.from_pretrained("....")

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size=128)

config = RobertaSmilesConfig()
model = RobertaSmiles(config)
optimizer = optim.Adam(model.parameters(), lr=1e-05, betas=(0.9, 0.98), eps=1e-09)
num_train_epochs = 300

#Train
for epoch in range(num_train_epochs):

    model.train()
    epoch_loss = []
    
    for batch in train_dataloader:
        optimizer.zero_grad() 
        batch = get_inputs_dict(batch)
        input_ids = batch['input_ids']
        smiles = batch['smiles']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, smiles=smiles, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        model.zero_grad()
        epoch_loss.append(loss.item())
        
    print('epoch',epoch,'Training avg loss',np.mean(epoch_loss))
    
    if (epoch % 30) == 0:
        torch.save(model,"...")

torch.save(model,"...")

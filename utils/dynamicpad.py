'''
Topic: Master thesis
Description: Class that creates the dataset with 
             dynamic padding for finetuning 

Created on: 16 February 2022
Created by: Jacopo Rizzo
'''
import torch

class DynamicPadDataset():

    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.text = data['text']
        self.targets = data['label']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]

        inputs = self.tokenizer(text, truncation = True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type = inputs['token_type_ids']

        return {'input_ids': torch.tensor(ids, dtype = torch.long),
                'attention_mask': torch.tensor(mask, dtype = torch.long),
                'token_type_ids': torch.tensor(token_type, dtype = torch.long),
                'label': torch.tensor(self.targets[index], dtype = torch.float64)}
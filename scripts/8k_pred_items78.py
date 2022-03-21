'''
Transfer learning on items 7 and 8 of
8ks.
'''
import pickle
import nltk
#nltk.download('punkt')
import nltk.data
#import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline)

# Load split 8k data
data = pickle.load(open('data/splitted_8k.pkl', 'rb'))

# Convert data and extract only items 7 and 8,
# i.e. according to our link 6 and 17. Save df.
df = data.to_pandas()

idx_label = []
for i in df['label']:
    idx_label.append(np.where(i == 1)[0][0])

df['idx_label'] = idx_label
df_only78 = df[(df['idx_label'] == 6) | (df['idx_label'] == 17)]
df_only78 = df_only78.reset_index(drop = True)
df_inf = Dataset.from_pandas(df_only78)

with open('data/split_8k_only78.pkl', 'wb') as fp:
    pickle.dump(df_inf, fp)

# Load model and tokenizer
model_name = 'bert-base-cased'
model_path = 'results/bert_v7/checkpoint-3560'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
#model = model.cuda()
#model.eval()

# Make predictions with pipeline
classifier = pipeline('text-classification', model = model, return_all_scores = True, tokenizer = tokenizer, truncation = True, function_to_apply = 'sigmoid', device = 0)

#with torch.no_grad():
prediction = classifier(df_inf['text'])

# Save predictions
with open('data/predictions_8k_items78.pkl', 'wb') as fp:
    pickle.dump(prediction, fp)
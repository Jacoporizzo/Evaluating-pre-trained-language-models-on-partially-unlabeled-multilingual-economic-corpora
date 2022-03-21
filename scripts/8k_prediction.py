'''
Transfer learning on the forms 8k data on the 
fine-tuned model, i.e. models. 
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

# Load and prepare the dataset
data = pickle.load(open('data/8k_text_gerclass.pkl', 'rb'))
splitter = nltk.data.load('tokenizers/punkt/english.pickle')

split_txt = []
for txt in data['text']:
    split_txt.append(splitter.tokenize(txt))

df = pd.DataFrame({'hash': data['hash'], 
                   'text': split_txt, 
                   'label': data['label']})

# Save each sentence singularly with document label
has = []
sents = []
lab_items = []
for form in range(len(df)):
    flatten = [item for item in df['text'][form]]
    sents.extend(flatten)
    lab_items.extend([df['label'][form]] * len(flatten))
    has.extend([df['hash'][form]] * len(flatten))

# Create df, remove rows with sentenece length < 2 and save as Dataset
df_sent_split = pd.DataFrame({'hash': has, 
                              'text': sents, 
                              'label': lab_items})
df_sent_split['length'] = [len(i) for i in df_sent_split['text']]
df_sent_split = df_sent_split[df_sent_split['length'] > 2]
df_sent_split = df_sent_split.reset_index(drop = True)
df_sent_split.drop('length', axis = 1, inplace = True)
df_split = Dataset.from_pandas(df_sent_split)

with open('data/splitted_8k.pkl','wb') as fp:
    pickle.dump(df_split, fp)

# Remove documents with item 7 and 8 for the first step and save. Removed
# according to index of labels_names (from Helper class).
idx_label = []
for i in df_sent_split['label']:
    idx_label.append(i.index(1))

df_sent_split['idx_label'] = idx_label
df_wo78 = df_sent_split[(df_sent_split['idx_label'] != 6) & (df_sent_split['idx_label'] != 17)]
df_wo78 = df_wo78.reset_index(drop = True)
df_inf = Dataset.from_pandas(df_wo78)

with open('data/8k_wo78.pkl', 'wb') as fp:
    pickle.dump(df_inf, fp)

# Load model and tokenizer
model_name = 'bert-base-cased'
model_path = 'results/bert_v7/checkpoint-3560'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
#model = model.cuda()
#model.eval()

# Make predictions with pipeline (for data w/o Items 7 and 8)
classifier = pipeline('text-classification', model = model, return_all_scores = True, tokenizer = tokenizer, truncation = True, function_to_apply = 'sigmoid', device = 0)

#with torch.no_grad():
prediction = classifier(df_inf['text'])

# Save predictions
with open('data/predictions_8k.pkl', 'wb') as fp:
    pickle.dump(prediction, fp)
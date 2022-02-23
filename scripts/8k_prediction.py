'''
Transfer learning on the forms 8k data on the 
fine-tuned model, i.e. models. 
'''
import pickle
import nltk
nltk.download('punkt')
import nltk.data
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

df = pd.DataFrame({'text': split_txt, 'label': data['label']})

# Save each sentence singularly with document label
sents = []
lab_items = []
for form in range(len(df)):
    flatten = [item for item in df['text'][form]]
    sents.extend(flatten)
    lab_items.extend([df['label'][form]] * len(flatten))

# Create df, remove rows with sentenece length == 1 and save as Dataset
df_sent_split = pd.DataFrame({'text': sents, 'label': lab_items})
df_sent_split = df_sent_split[df_sent_split['text'].str.split().str.len().ge(2)]
df_sent_split = df_sent_split.reset_index(drop = True)
df_split = Dataset.from_pandas(df_sent_split)

with open('data/split_8k.pkl','wb') as fp:
    pickle.dump(df_split, fp)

# Remove documents with item 7 and 8 for the first step and save. Removed
# according to index of labels_names (from Helper class).
idx_label = []
for i in df_sent_split['label']:
    idx_label.append(np.where(i == 1)[0][0])

df_sent_split['idx_label'] = idx_label
df_wo78 = df_sent_split[(df_sent_split['label'] != 6) & (df_sent_split['label'] != 17)]
df_wo78 = df_wo78.reset_index(drop = True)
df_inf = Dataset.from_pandas(df_wo78)

with open('data/split_8k_wo78.pkl', 'wb') as fp:
    pickle.dump(df_inf, fp)

# Load model and tokenizer
model_name = 'bert-base-cased'
model_path = 'results/checkpoint-10680_v5'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Make predictions with pipeline
classifier = pipeline('text-classification', model = model, return_all_scores = True, tokenizer = tokenizer, function_to_apply = 'sigmoid')
prediction = classifier(df_inf['text'])

# Save predictions
with open('predictions_8k_v5.pkl', 'wb') as fp:
    pickle.dump(prediction, fp)
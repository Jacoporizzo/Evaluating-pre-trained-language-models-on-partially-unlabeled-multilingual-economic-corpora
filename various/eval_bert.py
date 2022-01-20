'''
Evaluation of the first version of the 
finetuned BERT model on a portion of the data.
'''
import pickle
from utils.helpers import Helper
import torch
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline)

# Import data and model path
data = pickle.load(open('data/data_split_v1.pkl', 'rb'))
model_path = 'results/checkpoint-10680/'

# Import tokenizer and load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Get test data
test_df = data['test']
labels = torch.tensor(test_df['label'])

# Make predictions with pipeline
classifier = pipeline("text-classification", model = model, return_all_scores = True, tokenizer = tokenizer)
prediction = classifier(test_df['text'])

# Make predictions using the model (weird results)
# outputs = model(test_df['input_ids'], labels = labels.float())
# soft = torch.nn.Softmax(dim = 1)
# preds = soft(outputs.logits)
# labs_pred = torch.argmax(preds, dim = 1)

# Evaluate results
helper = Helper()

true_labels = helper.actual_labels(test_df['label'])
predicted_labels_scores = helper.predicted_labels_score(true_labels, prediction)
predicted_labels = helper.predicted_labels(predicted_labels_scores)

# Comparison between true_labels and predicted_labels
scores = helper.evaluation_scores(true_labels, predicted_labels)

# # For saved df
# true = pickle.load(open('data/true', 'rb'))
# pred = pickle.load(open('data/predicted', 'rb'))

# predicted = helper.predicted_labels(pred)

# helper.evaluation_scores(true, predicted, eval_schema = 'macro')

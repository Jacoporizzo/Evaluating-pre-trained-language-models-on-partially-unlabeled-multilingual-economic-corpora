'''
Evaluation of the first version of the 
finetuned BERT model on a portion of the data.
'''
import pickle
from utils.helpers import Helper
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline)

# Import data and model path
data = pickle.load(open('data/df_finetune.pkl', 'rb'))
model_path = 'results/checkpoint-10680/'

# Split dataset into train and test and conversion 
# to right format. Dataset split: train 80%, test 20%
df = Dataset.from_pandas(data).train_test_split(0.2, 0.8)

# Import tokenizer and load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Get test data
test_df = df['test']
test_df.set_format(type = 'torch')

# Make predictions
classifier = pipeline("text-classification", model = model, return_all_scores = True, tokenizer = tokenizer)
prediction = classifier(test_df['text'])

# Evaluate results
helper = Helper()
true_labels = helper.actual_labels(test_df['label'])
predicted_labels_scores = helper.predicted_labels_score(true_labels, prediction)
predicted_labels = helper.predicted_labels(predicted_labels_scores)

# Comparison between true_labels and predicted_labels

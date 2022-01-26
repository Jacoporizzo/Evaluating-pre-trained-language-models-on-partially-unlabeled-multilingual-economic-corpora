'''
Here we fine-tune the BERT model, which we will
use to do the transfer-learning on the 8k-forms.
This script can also be used to fine-tune any 
other model, that is present on the huggingface
hub.
'''
import pickle
import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer)

# Import data
data = pickle.load(open('data/df_finetune.pkl', 'rb'))

# Preprocess data and split into train/test/dev (80/10/10) set using
# stratified sampling based on one-label classes, i.e.
# for mulitlabeled datapoints only the first class is selected
data['lab'] = [np.nonzero(idx)[0][0] for idx in data['label']]

train, test = train_test_split(data, test_size = 0.2, stratify = data['lab'])
dev, test = train_test_split(test, test_size = 0.5, stratify = test['lab'])

data.drop('lab', axis = 1, inplace = True)
train.drop('lab', axis = 1, inplace = True)
test.drop('lab', axis = 1, inplace = True)
dev.drop('lab', axis = 1, inplace = True)

train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)
dev = Dataset.from_pandas(dev)

# Model's name and paramaters and metrics
model_name = 'bert-base-cased'
training_args = TrainingArguments(
    output_dir = '../results',
    evaluation_strategy = 'steps',
    learning_rate = 4e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 12,
    weight_decay = 0.01,
    logging_dir = '../results/logs',
    logging_first_step = True,
    logging_strategy = 'epoch',
    save_strategy = 'epoch',
    save_total_limit = 1
)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(input):
    return tokenizer(input['text'], padding = 'max_length', truncation = True)

train_df = train.map(tokenize_function, batched = True)
test_df = test.map(tokenize_function, batched = True)
dev_df = dev.map(tokenize_function, batched = True)

# Save datasets
#with open('data/train_data.pkl', 'wb') as fp:
#    pickle.dump(train_df, fp)
#
#with open('data/test_data.pkl', 'wb') as fp:
#    pickle.dump(test_df, fp)
#
#with open('data/dev_data.pkl', 'wb') as fp:
#    pickle.dump(dev_df, fp)

# Set data to right format
train_df.set_format(type = 'torch', columns = ['input_ids', 'token_type_ids', 'attention_mask'], output_all_columns = True)
dev_df.set_format(type = 'torch', columns = ['input_ids', 'token_type_ids', 'attention_mask'], output_all_columns = True)

# Define accuracy metric
def compute_metric(preds):
    logits, labels = preds
    predictions = np.argmax(logits, axis = -1)
    ref_labels = np.argmax(labels, axis = -1)
    acc = accuracy_score(ref_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ref_labels, predictions, average = 'macro')
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Initialize model and start trainig (i.e. finetuning)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 22)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_df,
    eval_dataset = dev_df,
    compute_metrics = compute_metric
)

trainer.train()
trainer.evaluate()
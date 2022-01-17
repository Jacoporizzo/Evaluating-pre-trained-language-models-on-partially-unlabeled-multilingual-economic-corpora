'''
Here we fine-tune the BERT model, which we will
use to do the transfer-learning on the 8k-forms.
This script can also be used to fine-tune any 
other model, that is present on the huggingface
hub.
'''
import pickle
import numpy as np
from datasets import Dataset, load_metric
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer)

# Import data
data = pickle.load(open('data/df_finetune.pkl', 'rb'))

# Split dataset into train and test and conversion 
# to right format. Dataset split: train 80%, test 20%
df = Dataset.from_pandas(data).train_test_split(0.2, 0.8)

# Model's name and paramaters and metrics
model_name = 'bert-base-cased'
training_args = TrainingArguments(
    output_dir = '../results',
    evaluation_strategy = 'steps',
    learning_rate = 4e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 6,
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

tokenized_df = df.map(tokenize_function, batched = True)

# Prepare data with DataLoader
train_df = tokenized_df['train']
test_df = tokenized_df['test']

#train_df.set_format(type = 'torch', columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label'])
#test_df.set_format(type = 'torch', columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label'])

# Define accuracy metric
def compute_metric(eval_pred):
    metric = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    ref_label = np.argmax(labels, axis = -1)
    return metric.compute(predictions=predictions, references=ref_label)

# Initialize model and start trainig (i.e. finetuning)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 22)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_df,
    eval_dataset = test_df,
    compute_metrics = compute_metric
)

trainer.train()
model.save_pretrained('../models')
'''
Inference on the test data using the fientuned model.
'''
import pickle
import torch
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline)

# Import test data and model path
test_data = pickle.load(open('data/data_v5/test_data_v5.pkl', 'rb'))
model_path = 'results/checkpoint-10680_v5'

# Load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Get test data
labels = torch.tensor(test_data['label'])

# Make predictions with pipeline
classifier = pipeline("text-classification", model = model, return_all_scores = True, tokenizer = tokenizer, function_to_apply = 'sigmoid')
prediction = classifier(test_data['text'])

# Save predictions
with open('predictions_test_v5.pkl', 'wb') as fp:
    pickle.dump(prediction, fp)

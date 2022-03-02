'''
Inference on the test data using the finetuned model.
'''
import pickle
#import torch
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline)

# Import test data and model path
test_data = pickle.load(open('data/data_v7/test_data.pkl', 'rb'))
model_path = 'results/bert_v7/checkpoint-3560'

# Load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained(model_path)
#model = model.cuda()
#model.eval()

# Make predictions with pipeline
classifier = pipeline("text-classification", model = model, return_all_scores = True, tokenizer = tokenizer, function_to_apply = 'sigmoid', device = 0)
#with torch.no_grad():
prediction = classifier(test_data['text'])

# Save predictions
with open('prediction_test.pkl', 'wb') as fp:
    pickle.dump(prediction, fp)

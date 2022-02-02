'''
Inference on the dataset conatianing the fulltext.
Since some inputs are longer than 512 tokens, these have 
been truncated to have this length.
'''
import pickle
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline)

# Import test data and model path
test_data = pickle.load(open('data/labelled_doc.pkl', 'rb'))
model_path = 'results/checkpoint-10680_v5'

# Load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Make predictions with pipeline
classifier = pipeline("text-classification", model = model, return_all_scores = True, tokenizer = tokenizer, function_to_apply = 'sigmoid')
prediction = classifier(list(test_data['text']), padding = True, truncation = True)

# save predictions
with open('data/data_v5/prediction_fulltext_v5.pkl', 'wb') as fp:
    pickle.dump(prediction, fp)
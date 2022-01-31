'''
Evaluation of the predictions of the 
finetuned BERT model on the test data.
'''
import pickle
from utils.helpers import Helper

# Import test data and predictions
test_data = pickle.load(open('data/data_v5/test_data_v5.pkl', 'rb'))
predictions = pickle.load(open('data/data_v5/prediction_test_v5.pkl', 'rb'))

# Evaluate results
helper = Helper()

# Get true labels of test data
true_labels = helper.actual_labels(test_data['label'])

# Get predicted labels for test data
predicted_labels_scores = helper.predicted_labels_scores(true_labels, predictions)
predicted_labels = helper.predicted_labels(predicted_labels_scores)

# Compute geeneral evaluation metrics
scores_global = helper.evaluation_scores(true_labels, predicted_labels)
scores_local = helper.evaluation_scores(true_labels, predicted_labels, level = 'local')

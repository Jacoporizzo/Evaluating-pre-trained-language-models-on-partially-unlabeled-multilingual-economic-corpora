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

# Compute evaluation metrics
scores_global = helper.evaluation_scores(true_labels, predicted_labels)
scores_local = helper.evaluation_scores(true_labels, predicted_labels, level = 'local')

#%%
'''
Evalaution of predictions for fulltexts' predictions.
'''
# Compute inference using entire text of labelled documents
fulltext = pickle.load(open('data/labelled_doc.pkl', 'rb'))
ft_preds = pickle.load(open('data/data_v5/prediction_fulltext_v5.pkl', 'rb'))

# Evaluate results
helper = Helper()

# Get true labels 
ft_true = helper.actual_labels(fulltext['labels'])

# Get predicted labels
ft_predicted_ls = helper.predicted_labels_scores(ft_true, ft_preds)
ft_predicted = helper.predicted_labels(ft_predicted_ls)

# Compute evaluation metrics
ft_global = helper.evaluation_scores(ft_true, ft_predicted)
ft_local = helper.evaluation_scores(ft_true, ft_predicted, level = 'local')

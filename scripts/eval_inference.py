'''
Evaluation of the predictions of the 
finetuned BERT model on the test data.
'''
import pickle
from utils.helpers import Helper
from utils.doclevel import DocLevel

### Evaluation on sentence level
# Import test data and predictions
test_data = pickle.load(open('data/data_v7/test_data.pkl', 'rb'))
predictions = pickle.load(open('data/data_v7/prediction_test_fourthepoch.pkl', 'rb'))

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

### Evaluation on document level
docs = DocLevel()

document_labels = docs.doc_labels(test_data)
document_predictions = docs.doc_predictions(test_data, predictions)

document_cls = docs.remove_empty_class(document_labels, document_predictions)

doc_local_eval = docs.doc_evaluations(document_cls['label_true'], document_cls['label_predicted'], 'local')
doc_global_eval = docs.doc_evaluations(document_cls['label_true'], document_cls['label_predicted'])

'''
Evaluation of the predictions for the 8k-forms
'''
import pickle
from utils.helpers import Helper
from utils.doclevel import DocLevel

### Evaluation on sentence level
# Import 8k data and predictions
data = pickle.load(open('data/8k_wo78.pkl', 'rb'))
predictions = pickle.load(open('data/predictions_8k.pkl', 'rb'))

# Evaluate results
helper = Helper()

# Get true labels of 8k (i.e. linked true labels)
true_labels = helper.actual_labels(data['label'])

# Get predicted labels for 8k
predicted_labels_scores = helper.predicted_labels_scores(true_labels, predictions)
predicted_labels = helper.predicted_labels(predicted_labels_scores)

# Compute evaluation metrics
scrs_global = helper.evaluation_scores(true_labels, predicted_labels)
scrs_local = helper.evaluation_scores(true_labels, predicted_labels, level = 'local')

### Evaluation on document level
docs = DocLevel()

items_labels = docs.labels_8k(data)
items_predictions = docs.predictions_8k(data, predictions)

items_cls = docs.remove_empty_class(items_labels, items_predictions)

items_local = docs.doc_evaluations(items_cls['label_true'], items_cls['label_predicted'], 'local')
items_global = docs.doc_evaluations(items_cls['label_true'], items_cls['label_predicted'])

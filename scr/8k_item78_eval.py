'''
Evaluation of predictions for items 7 and 8
'''
import pickle
from utils.helpers import Helper
from utils.doclevel import DocLevel
import plotnine as pn
import pandas as pd
import numpy as np

# Import predictions
data = pickle.load(open('data/split_8k_only78.pkl', 'rb'))
predictions = pickle.load(open('data/predictions_8k_items78.pkl', 'rb'))

helper = Helper()

true_labels = helper.actual_labels(data['label'])

predicted_thr = helper.threshold_classification(predictions, threshold = 0.4)
predicted = helper.predicted_labels(predicted_thr)

# Plot predicted labels
lab_names = helper.get_labels_names()
dicts = dict(zip(range(0,22), lab_names))

true_names = []
for i in true_labels:
    for key in dicts:
        if key in i:
            true_names.append(dicts[key])

predicted_names = []
for i in predicted:
    if len(i) == 0:
        predicted_names.append(['NC'])
    elif len(i) == 1:
        for j in i:
            predicted_names.append([dicts[j]])
    else:
        multi = []
        for t in i:
            multi.append(dicts[t])
        predicted_names.append(multi)
        
df = pd.DataFrame({'true': [item for item in true_labels for item in item],
                   'prediction': predicted,
                   'true_labels': true_names,
                   'prediction_labels': predicted_names})

dividende = df[df["true"] == 6].reset_index(drop=True)
ruckkauf = df[df['true'] == 17].reset_index(drop=True)

dividende_pred = [item for items in dividende['prediction_labels'] for item in items]
dividende_sum = pd.DataFrame.from_dict(dict([[x, dividende_pred.count(x)] for x in set(dividende_pred)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
dividende_sum['proportion'] = dividende_sum['total']/(sum(dividende_sum['total']))

ruckkauf_pred = [item for items in ruckkauf['prediction_labels'] for item in items]
ruckkauf_sum = pd.DataFrame.from_dict(dict([[x, ruckkauf_pred.count(x)] for x in set(ruckkauf_pred)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
ruckkauf_sum['proportion'] = ruckkauf_sum['total']/(sum(ruckkauf_sum['total']))

# Plot for Dividende class (item 7) prediction
(pn.ggplot(dividende_sum, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "Predicted labels for items 7")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Plot for Ruckkauf class (item 8) prediction
(pn.ggplot(ruckkauf_sum, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "Predicted labels for items 8")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

#----------------------------------
## Document level
docs = DocLevel()

doc_labels = docs.labels_8k(data)
doc_predictions = docs.predictions_8k(data, predictions, threshold = 0.4)

doc_cls = docs.remove_empty_class(doc_labels, doc_predictions)

doc_true = [np.nonzero(doc_cls['label_true'][i])[0][0] for i in range(len(doc_cls))]

doc_pred = [np.nonzero(doc_cls['label_predicted'][j]) for j in range(len(doc_cls))]
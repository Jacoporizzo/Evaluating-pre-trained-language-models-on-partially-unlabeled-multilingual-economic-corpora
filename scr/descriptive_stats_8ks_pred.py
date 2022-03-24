import pickle
import pandas as pd
import plotnine as pn
from utils.helpers import Helper
from utils.doclevel import DocLevel

# Import data and prediction
data = pickle.load(open('data/8k_wo78.pkl', 'rb'))
predictions = pickle.load(open('data/predictions_8k.pkl', 'rb'))

#---------------------------------------------------------------
### Sentence level
## Plot distribution of linked labels for true data (8k_true_sent_relfreq)
df = data.data.to_pandas()

helper = Helper()
lab_names = helper.get_labels_names()
dicts = dict(zip(range(0,22), lab_names))

true_labels = helper.actual_labels(data['label'])

true_names = []
for i in true_labels:
    for key in dicts:
        if key in i:
            true_names.append(dicts[key])
            
# Get summary of true labels
labs_df = pd.DataFrame.from_dict(dict([[x, true_names.count(x)] for x in set(true_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
labs_df['proportion'] = labs_df['total']/sum(labs_df['total'])

(pn.ggplot(labs_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "8ks' true class on sentence level")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

## Plot distribution of predicted labels (8k_pred_sent_relfreq)
preds = helper.threshold_classification(predictions, threshold = 0.4)
pred_labels = helper.predicted_labels(preds)
    
# Save labels' names
pred_names = []
for i in pred_labels:
    if len(i) == 0:
        pred_names.append(['NC'])
    elif len(i) == 1:
        for j in i:
            pred_names.append([dicts[j]])
    else:
        multi = []
        for t in i:
            multi.append(dicts[t])
        pred_names.append(multi)

predicted_names = [item for items in pred_names for item in items]
pred_labs_df = pd.DataFrame.from_dict(dict([[x, predicted_names.count(x)] for x in set(predicted_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
pred_labs_df['proportion'] = pred_labs_df['total']/(sum(pred_labs_df['total']))

(pn.ggplot(pred_labs_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "8ks' predicted class on sentence level")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Document level
docs = DocLevel()

items_labels = docs.labels_8k(data)
items_predictions = docs.predictions_8k(data, predictions, threshold = 0.4)

items_cls = docs.remove_empty_class(items_labels, items_predictions)

doc_pred = []
for j in range(len(items_cls)):
    arr = np.nonzero(items_cls['label_predicted'][j])
    multi = []
    for i in range(len(arr[0])):
        multi.append(arr[0][i])
    doc_pred.append(multi)
    
doc_predicted_names = []
for i in doc_pred:
    if len(i) == 0:
        doc_predicted_names.append(['NC'])
    elif len(i) == 1:
        for j in i:
            doc_predicted_names.append([dicts[j]])
    else:
        multi = []
        for t in i:
            multi.append(dicts[t])
        doc_predicted_names.append(multi)
        
doc_pred_names = [item for items in doc_predicted_names for item in items]
doc_labs_df = pd.DataFrame.from_dict(dict([[x, doc_pred_names.count(x)] for x in set(doc_pred_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
doc_labs_df['proportion'] = doc_labs_df['total']/(sum(doc_labs_df['total']))

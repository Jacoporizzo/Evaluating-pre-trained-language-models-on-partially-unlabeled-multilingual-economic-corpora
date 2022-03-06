import pickle
import pandas as pd
import plotnine as pn
from utils.helpers import Helper

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

test_labels = []
for lab in df['label']:
    test_labels.append([idx for idx in range(len(lab)) if lab[idx] == 1])

test_names = []
for i in test_labels:
    for key in dicts:
        if key in i:
            test_names.append(dicts[key])
            
# Get summary of true labels
labs_df = pd.DataFrame.from_dict(dict([[x, test_names.count(x)] for x in set(test_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
labs_df['proportion'] = labs_df['total']/sum(labs_df['total'])

(pn.ggplot(labs_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "8ks' true class on sentence level")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

## Plot distribution of predicted labels (8k_pred_sent_relfreq)
preds = helper.predicted_labels_scores(test_labels, predictions)
pred_labels = helper.predicted_labels(preds)
    
# Save labels' names
pred_names = []
for i in pred_labels:
    for key in dicts:
        if key in i:
            pred_names.append(dicts[key])

pred_labs_df = pd.DataFrame.from_dict(dict([[x, pred_names.count(x)] for x in set(pred_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
pred_labs_df['proportion'] = pred_labs_df['total']/(sum(pred_labs_df['total']))

(pn.ggplot(pred_labs_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "8ks' predicted class on sentence level")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))
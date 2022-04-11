'''
Evaluation of predictions for items 7 and 8
'''
import pickle
from utils.helpers import Helper
from utils.doclevel import DocLevel
from utils.plots import PlotData
from random import randint
import plotnine as pn
import pandas as pd
import numpy as np

# Import predictions
data = pickle.load(open('data/8k_split_only78.pkl', 'rb'))
predictions = pickle.load(open('data/predictions_8k_items78_thirdepoch.pkl', 'rb'))

helper = Helper()
plot = PlotData()

true_labels = helper.actual_labels(data['label'])

true_names = plot.labels_names(true_labels)

predicted_thr = helper.threshold_classification(predictions, threshold = 0.45)
predicted = helper.predicted_labels(predicted_thr)

pred_names = plot.labels_names(predicted)

# Plot predicted labels
df = pd.DataFrame({'true': [item for item in true_labels for item in item],
                   'prediction': predicted,
                   'true_labels': true_names,
                   'prediction_labels': pred_names})

dividende = df[df["true"] == 6].reset_index(drop=True)
ruckkauf = df[df['true'] == 17].reset_index(drop=True)

divids = plot.labels_complete_df(dividende['prediction_labels'])
rucks = plot.labels_complete_df(ruckkauf['prediction_labels'])

# Plot for Dividende class (item 7) prediction
(pn.ggplot(divids, pn.aes(x = 'label', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "Predicted labels for items 7")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Plot for Ruckkauf class (item 8) prediction
(pn.ggplot(rucks, pn.aes(x = 'label', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "Predicted labels for items 8")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

scrs_local = helper.evaluation_scores(true_labels, predicted, level = 'local')

#----------------------------------
## Document level
docs = DocLevel()

doc_labels = docs.labels_8k(data)
doc_predictions = docs.predictions_8k(data, predictions, threshold = 0.45)

# Remove irrelevant class. ATTENTION: Output "irrelevant" is the "empty" class
doc_cls = docs.remove_irrelevant_class(doc_labels, doc_predictions)

items_scrs = docs.doc_evaluations(doc_cls['label_true'], doc_cls['label_predicted'], level = 'local')

doc_true_int = plot.labels_doc(doc_cls)
doc_pred_int = plot.labels_doc(doc_cls, prediction = True)
        
doc_true_names = plot.labels_names(doc_true_int)
doc_pred_names = plot.labels_names(doc_pred_int)

doc_df = pd.DataFrame({'true': [item for item in doc_true_int for item in item],
                       'prediction': doc_pred_int,
                       'true_labels': doc_true_names,
                       'prediction_labels': doc_pred_names})

doc_dividende = doc_df[doc_df["true"] == 6].reset_index(drop=True)
doc_ruckkauf = doc_df[doc_df['true'] == 17].reset_index(drop=True)

doc_divids = plot.labels_complete_df(doc_dividende['prediction_labels'])
doc_rucks = plot.labels_complete_df(doc_ruckkauf['prediction_labels'])

doc_divids = doc_divids.replace('Irrelevant', 'Empty')
doc_rucks = doc_rucks.replace('Irrelevant', 'Empty')

# Prepare dataset to plot
doc_divids['class'] = 'Dividende'
doc_rucks['class'] = 'Rückkauf'

doc_divids = doc_divids.replace({'Pharma_Good': 'Pharma Good', 'Real_Invest': 'Real Invest'})
doc_rucks = doc_rucks.replace({'Pharma_Good': 'Pharma Good', 'Real_Invest': 'Real Invest'})

category =  ['Earnings',
             'SEO',
             'Management',
             'Guidance',
             'Gewinnwarnung',
             'Beteiligung',
             'Dividende',
             'Restructuring',
             'Debt',
             'Law',
             'Großauftrag',
             'Squeeze',
             'Insolvenzantrag',
             'Insolvenzplan',
             'Delay',
             'Split',
             'Pharma Good',
             'Rückkauf',
             'Real Invest',
             'Delisting',
             'Empty']

doc_divids['label'] = pd.Categorical(doc_divids['label'], categories = category)
doc_rucks['label'] = pd.Categorical(doc_rucks['label'], categories = category)

color = []
for i in range(21):
    color.append('#%06X' % randint(0, 0xFFFFFF))

(pn.ggplot(doc_divids, pn.aes(x = 'class', y = 'proportion', fill = 'label'))
     + pn.geom_bar(position = 'fill', stat = 'identity')
     + pn.geom_bar(mapping = pn.aes(x = 'class', y = 'proportion', fill = 'label'), 
                   data = doc_rucks, position = 'fill', stat = 'identity')
     + pn.labs(x = '', y = 'Relative frequency', title = 'Predictions for items 7 and 8')
     + pn.guides(fill = pn.guide_legend(title = ''))
     + pn.scale_fill_manual(values = color))
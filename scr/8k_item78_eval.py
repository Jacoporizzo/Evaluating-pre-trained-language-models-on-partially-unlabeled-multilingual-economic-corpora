'''
Evaluation of predictions for items 7 and 8
'''
import pickle
from utils.helpers import Helper
from utils.doclevel import DocLevel
from utils.plots import PlotData
import plotnine as pn
import pandas as pd
import numpy as np

# Import predictions
data = pickle.load(open('data/split_8k_only78.pkl', 'rb'))
predictions = pickle.load(open('data/predictions_8k_items78.pkl', 'rb'))

helper = Helper()
plot = PlotData()

true_labels = helper.actual_labels(data['label'])

true_names = plot.labels_names(true_labels)

predicted_thr = helper.threshold_classification(predictions, threshold = 0.4)
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

#----------------------------------
## Document level
docs = DocLevel()

doc_labels = docs.labels_8k(data)
doc_predictions = docs.predictions_8k(data, predictions, threshold = 0.4)

# Remove irrelevant class. ATTENTION: Output "irrelevant" is the "empty" class
doc_cls = docs.remove_irrelevant_class(doc_labels, doc_predictions)

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

'''
Descriptive statistics for the 8k predictions
'''
import pickle
import plotnine as pn
from utils.helpers import Helper
from utils.doclevel import DocLevel
from utils.plots import PlotData

# Import data and prediction
data = pickle.load(open('data/8k_wo78.pkl', 'rb'))
predictions = pickle.load(open('data/predictions_8k.pkl', 'rb'))

#---------------------------------------------------------------
### Sentence level
## Plot distribution of linked labels for true data (8k_true_sent_relfreq)
df = data.data.to_pandas()

helper = Helper()
plot = PlotData() 

true_labels = helper.actual_labels(data['label'])
          
# Get summary of true labels
true_names = plot.labels_names(true_labels)
sentence_labs = plot.labels_complete_df(true_names)

(pn.ggplot(sentence_labs, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "8ks' true class on sentence level")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

## Plot distribution of predicted labels (8k_pred_sent_relfreq)
preds = helper.threshold_classification(predictions, threshold = 0.4)
pred_labels = helper.predicted_labels(preds)
    
# Save labels' names
pred_names = plot.labels_names(pred_labels)
pred_sentence_labs = plot.labels_complete_df(pred_names)

(pn.ggplot(pred_sentence_labs, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "8ks' predicted class on sentence level")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Document level
docs = DocLevel()

items_labels = docs.labels_8k(data)
items_predictions = docs.predictions_8k(data, predictions, threshold = 0.4)

items_cls = docs.remove_irrelevant_class(items_labels, items_predictions)

doc_pred = plot.labels_doc(items_cls, prediction = True)
doc_predicted_names = plot.labels_names(doc_pred)

doc_labs_df = plot.labels_complete_df(doc_predicted_names)

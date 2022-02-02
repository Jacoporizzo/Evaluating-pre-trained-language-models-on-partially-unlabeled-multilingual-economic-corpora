'''
Descriptive statistics for the orginal data  
and for the results of bert v5.
'''
import json
import pickle
import pandas as pd
import plotnine as pn
from utils.imports import Import
from utils.helpers import Helper

# Import data
imports = Import()
gold_data = imports.importgold()

# Bar chart of the labels (absolute frequency).
# The amount and distribution of the labels is 
# the same for the goldstandards and the SBERT's output.
labels = gold_data.iloc[:,4:26]
bars = labels.sum(axis = 0)
bc = bars.plot.bar()

# Same with ggplot
bars_df = bars.reset_index().rename(columns = {0: 'total'})
(pn.ggplot(bars_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Total labels', title = 'Absolute frequency of true labels in entire dataset')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Import trainer state data and extract relevant info
eval_data = json.load(open('results/checkpoint-10680_v5/trainer_state.json', 'r'))
performance = eval_data['log_history']

# Split into evaluation metrics and general info
metrics = [dic for dic in performance if len(dic) == 10]
general = [dic for dic in performance if len(dic) == 4]

# Convert to dataframes
df_metrics = pd.DataFrame(metrics)
df_general = pd.DataFrame(general)

# Plot
(pn.ggplot(df_metrics, pn.aes(x = 'epoch', y = 'eval_accuracy'))
     + pn.geom_line()
     + pn.xlim(1,12)
     + pn.ylim(0.68, 0.76)
     + pn.geom_point()
     + pn.scale_x_continuous(name="Epoch", limits=[1, 12], 
                             breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
     + pn.labs(y = 'Accuracy', title = 'Accuracy on evaluation dataset per epoch'))

# Pyplot version
# plt.plot('epoch', 'eval_accuracy', data = df_metrics)
# plt.show()

# Barplot for test_data's labels
test_data = pickle.load(open('data/data_v5/test_data_v5.pkl', 'rb'))
test_labels = []
for lab in test_data['label']:
    test_labels.append([idx for idx in range(len(lab)) if lab[idx] == 1])
    
helper = Helper()
lab_names = helper.get_labels_names()

dicts = dict(zip(range(0,22), lab_names))
    
# Save labels' names
test_names = []
for i in test_labels:
    for key in dicts:
        if key in i:
            test_names.append(dicts[key])

# Plotting with matplotlib            
# keys, counts = np.unique(test_names, return_counts = True)
# plt.bar(keys, counts)
# plt.show()

# Same with ggplot
labs_df = pd.DataFrame.from_dict(dict([[x, test_names.count(x)] for x in set(test_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})

(pn.ggplot(labs_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Total labels', title = 'Absolute frequency of true labels in test dataset')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Absolute frequency of predicted labels on test dataset
test_pred = pickle.load(open('data/data_v5/prediction_test_v5.pkl', 'rb'))
preds = helper.predicted_labels_scores(test_labels, test_pred)
pred_labels = helper.predicted_labels(preds)
    
# Save labels' names
pred_names = []
for i in pred_labels:
    for key in dicts:
        if key in i:
            pred_names.append(dicts[key])

# Same with ggplot
pred_labs_df = pd.DataFrame.from_dict(dict([[x, pred_names.count(x)] for x in set(pred_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})

(pn.ggplot(pred_labs_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Total labels', title = 'Absolute frequency of predicted labels in test dataset')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Compute evaluation metrics for the test data 
local_metrics = helper.evaluation_scores(test_labels, pred_labels, 'local')
global_metrics = helper.evaluation_scores(test_labels, pred_labels)

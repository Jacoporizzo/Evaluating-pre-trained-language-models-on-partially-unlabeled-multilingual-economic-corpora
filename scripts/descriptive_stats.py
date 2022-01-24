'''
Descriptive statistics for the orginal data.
'''
import json
import pickle
import pandas as pd
import plotnine as pn
from utils.imports import Import
from utils.helpers import Helper
import matplotlib.pyplot as plt

# Import data
imports = Import()
gold_data = imports.importgold()

# Bar chart of the labels (absolute frequency).
# The amount and distribution of the labels is 
# the same for the goldstandards and the SBERT's output.
labels = gold_data.iloc[:,4:26]
bars = labels.sum(axis = 0)
bc = bars.plot.bar()

# Import trainer state data and extract relevant info
eval_data = json.load(open('results/checkpoint-10680/trainer_state.json', 'r'))
performance = eval_data['log_history']

# Split into evaluation metrics and general info
metrics = [dic for dic in performance if len(dic) == 10]
general = [dic for dic in performance if len(dic) == 4]

# Convert to dataframes
df_metrics = pd.DataFrame(metrics)
df_general = pd.DataFrame(general)

# Plot
(pn.ggplot(df_metrics)
     + pn.aes(x = 'epoch', y = 'eval_accuracy')
     + pn.geom_line()
     + pn.xlim(1,12)
     + pn.geom_point()
     + pn.scale_x_continuous(name="Epoch", limits=[1, 12], 
                             breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
     + pn.labs(y = 'Accuracy', title = 'Accuracy on evaluation dataset per epoch'))

# Pyplot version
plt.plot('epoch', 'eval_accuracy', data = df_metrics)
plt.show()

# Barplot for test_data's labels
training_data = pickle.load(open('data/data_split_v1.pkl', 'rb'))
test_data = training_data['test']
test_labels = []
for lab in test_data['label']:
    test_labels.append([idx for idx in range(len(lab)) if lab[idx] == 1])
    
helper = Helper()
lab_names = helper.get_labels_names()

dicts = dict(zip(range(0,22), lab_names))

names = []
for lab_nr in test_labels[0:5]:
    if len(lab_nr) == 1:
        names.append([dicts[lab_nr][0]])
    else:
        lst_labs = []
        length = len(lab_nr)
        for length in :
            
 for i in range(len(actual)):
    n_labels = len(actual[i])
    sort_pred = sorted(predicted[i], key = lambda x: x['score'], reverse = True)
    predicted_labels.append(sort_pred[0:n_labels])
    
# Find way to save labels' names in pandas dataframe
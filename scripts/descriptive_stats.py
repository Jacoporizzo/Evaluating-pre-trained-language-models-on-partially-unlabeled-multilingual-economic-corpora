'''
Descriptive statistics for the orginal data.
'''
import json
import pickle
import pandas as pd
import numpy as np
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

# Same with ggplot
bars_df = bars.reset_index().rename(columns = {0: 'total'})
(pn.ggplot(bars_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Total labels')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

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
(pn.ggplot(df_metrics, pn.aes(x = 'epoch', y = 'eval_accuracy'))
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
    
# Save labels' names
names = []
for i in test_labels:
    for key in dicts:
        if key in i:
            names.append(dicts[key])
            
keys, counts = np.unique(names, return_counts = True)

plt.bar(keys, counts)
plt.show()

labs_df = pd.DataFrame.from_dict(dict([[x, names.count(x)] for x in set(names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})

(pn.ggplot(labs_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.ylim(0, 1650)
     + pn.labs(y = 'Total labels')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

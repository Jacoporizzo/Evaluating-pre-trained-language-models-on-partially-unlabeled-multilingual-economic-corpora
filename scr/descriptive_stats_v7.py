'''
Descriptive statistics for the results and the
training process of the finetuned bert-base-cased V5
model.
'''
import json
import pickle
import pandas as pd
import plotnine as pn
from utils.helpers import Helper

# Import trainer state data and extract relevant info
eval_data = json.load(open('results/bert_v7/checkpoint-7120/trainer_state.json', 'r'))
performance = eval_data['log_history']

# Split into evaluation metrics and general info
metrics = [dic for dic in performance if len(dic) == 10]
general = [dic for dic in performance if len(dic) == 4]

# Convert to dataframes
df_metrics = pd.DataFrame(metrics)
df_general = pd.DataFrame(general)

# Plot accuracy over epochs (accuracy_testset)
(pn.ggplot(df_metrics, pn.aes(x = 'epoch', y = 'eval_accuracy'))
     + pn.geom_line()
     + pn.xlim(1,8)
     + pn.ylim(0.68, 0.76)
     + pn.geom_point()
     + pn.scale_x_continuous(name="Epoch", limits=[1, 8], 
                             breaks = [1, 2, 3, 4, 5, 6, 7, 8])
     + pn.labs(y = 'Accuracy', x = 'Epoch', title = 'Accuracy on evaluation dataset per epoch'))

# Validation and training loss (loss_trainevalset)
df_comparison = pd.DataFrame({'epoch': df_metrics['epoch'],
                              'eval_loss': df_metrics['eval_loss'],
                              'train_loss': df_general['loss'][1:].reset_index(drop=True)})


(pn.ggplot(df_comparison, pn.aes(x = 'epoch'))
     + pn.geom_line(pn.aes(y = 'eval_loss', color = '"blue"'))
     + pn.geom_line(pn.aes(y = 'train_loss', color = '"red"'))
     + pn.scale_x_continuous(name = 'Epoch', limits = [1, 8],
                           breaks = [1, 2, 3, 4, 5, 6, 7, 8])
     + pn.labs(y = 'Loss', x = 'Epoch', title = 'Train and validation loss per epoch')
     + pn.scale_colour_identity(guide = 'legend', name = 'Set',
                                breaks = ['red', 'blue'],
                                labels = ['Train', 'Validation']))

# Absolute and relative freq for test datas' true labels
test_data = pickle.load(open('data/data_v7/test_data.pkl', 'rb'))
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

labs_df = pd.DataFrame.from_dict(dict([[x, test_names.count(x)] for x in set(test_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
(pn.ggplot(labs_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Absolute frequency', x = '', title = 'True labels in test dataset')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

labs_df['proportion'] = labs_df['total']/sum(labs_df['total'])
(pn.ggplot(labs_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = 'True labels in test dataset')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

# Absolute and realtive freq for test datas' predicted labels
test_pred = pickle.load(open('data/data_v7/prediction_test_fourthepoch.pkl', 'rb'))
preds = helper.threshold_classification(test_pred, threshold = 0.4)
pred_labels = helper.predicted_labels(preds)
    
# Save labels' names
pred_names = []
for i in pred_labels:
    for key in dicts:
        if key in i:
            pred_names.append(dicts[key])

pred_labs_df = pd.DataFrame.from_dict(dict([[x, pred_names.count(x)] for x in set(pred_names)]), orient = 'index').reset_index().rename(columns = {0: 'total'})
(pn.ggplot(pred_labs_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Absolute frequency', x = '', title = 'Predicted labels in test dataset')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

pred_labs_df['proportion'] = pred_labs_df['total']/(sum(pred_labs_df['total']))
(pn.ggplot(pred_labs_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = 'Predicted labels in test dataset')
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))
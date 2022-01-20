'''
The goldstandards for 15 randomly selected Ad-Hocs are 
taken and "manually" translated. This manual translation 
is compared to the output of the SBERT model and evaluated 
using statistical measures, like precison, recall and 
F1-score. Adjust the path to the data location before running.
Note that the random selection has been previously made and
not reported, but the data are stored in a separate df.  
'''

from utils.evaluations import Evaluation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## ATTENTION: The following lines (18-33) are commented out and reported 
## only for the sake of clarity. These import and prepare the random
## selected documents. The df used for the evaluation, containing
## the prepared data is stored separately and is imported below.

#from utils.imports import Import
#from utils.translations import Translation
#from utils.labels import Label

## Import and extract data 
#imp = Import()
#data = imp.findcounterpart()
#goldstd = imp.importgold()
#random_data = pd.read_csv('data/random_docs.csv')
#rand_data = data[data['hash_y'].isin(random_data['hash_y'])].reset_index(drop=True)

## Pass data in model
#trans = Translation()
#label = Label()
#rand_cosines = trans.cosine_similarity(rand_data, goldstd)
#rand_labelled = label.merge_labels(rand_cosines, goldstd)

# Import the final df
manual_trans = pd.read_excel('data/Manual_labelling.xlsx', sheet_name = 'Sheet1')

# Get absolute and relative frequency of manual equals output
manual_trans['Manual_equals_output'].sum()
manual_trans['Manual_equals_output'].sum() / len(manual_trans['Manual_equals_output'])

# Create confusion matrix
evaluation = Evaluation()

equality_check = evaluation.is_equal(manual_trans)
confusion_matrix = evaluation.create_confusion(equality_check)

# Compute evaluations
true_positives = confusion_matrix['Predicted_1']['Actual_1']
false_positives = confusion_matrix['Predicted_1']['Actual_0']
false_negatives = confusion_matrix['Predicted_0']['Actual_1']

precision = true_positives / (true_positives + false_positives)

recall = true_positives / (true_positives + false_negatives)

f1_score = 2 * ((precision * recall) / (precision + recall))

print('Evaluation metrics for random data' + '\n' + 
      'Precision: {}'.format(round(precision, 4)) + '\n' +
      'Recall: {}'.format(round(recall, 4)) + '\n' +
      'F1-Score: {}'.format(round(f1_score, 4)))

# Plot heatmap for the confusion matrix
ax = plt.axes()
ax = sns.heatmap(confusion_matrix, ax = ax, annot = True,
                  annot_kws = {'size': 16}, cmap = 'rocket_r', fmt = 'g')
ax.set_title('Heatmap for random selected data')
plt.show()
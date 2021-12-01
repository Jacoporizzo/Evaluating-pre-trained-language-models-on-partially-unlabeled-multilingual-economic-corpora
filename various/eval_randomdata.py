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

## ATTENTION: The following lines (18-33) are commented out and reported 
## only for the sake of clarity. These imports and prepare the random
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

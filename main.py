# First way of setting up the df 
# import os
# os.chdir('C:/Users/Jacopo/Desktop/MA')
from utils.imports import Import
from utils.translations import Translation
from utils.labels import Label
from datetime import datetime
import pandas as pd

# Get execution time
start = datetime.now()

# Import german labeled ad-hoc and their english counterpart and goldstandards
imp = Import()
data = imp.findcounterpart()
goldstd = imp.importgold()

# Extract subdf of the 15 random selected documents, which has also
# been manually labelled
random_data = pd.read_csv('data/random_docs.csv')
test_data = data[data['hash_y'].isin(random_data['hash_y'])].reset_index(drop=True)

# Compute (cosine) similarity between goldstandards and english sentences
trans = Translation()
cosine_scores = trans.cosine_similarity(test_data, goldstd)

# Compute the english labeled dataframe basing on the cosine similarity outcome
label = Label()
english_labeled = label.merge_labels(cosine_scores, goldstd)

print(datetime.now() - start)

#%% Evaluation of results on random selected documents

# Import manually labelled dataset
manual = pd.read_excel('data/Manual_labelling.xlsx', sheet_name = 'Sheet1')

# Get absolute and relative frequency of manual equals output
manual['Manual_equals_output'].sum()
manual['Manual_equals_output'].sum() / len(manual['Manual_equals_output'])

# Create confusion matrix
df_conf = pd.DataFrame([manual['English_manual'], 
                        manual['English_cosine_output']]).T

df_conf.rename(columns = {'English_manual': 'Actual_translation', 
                          'English_cosine_output': 'Predicted_translation'}, inplace = True)

confusion_matrix = pd.crosstab(df_conf['Actual_translation'], 
                               df_conf['Predicted_translation'], 
                               rownames = ['Actual'], colnames = ['Predicted'])

#%% Import preprocessed data
import pickle 

english_gold = pickle.load(open('data/english_goldstandards.pkl', 'rb'))
cosines_scores = pickle.load(open('data/cosine_scores.pkl', 'rb'))

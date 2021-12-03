# First way of setting up the df 
# import os
# os.chdir('C:/Users/Jacopo/Desktop/MA')
from utils.imports import Import
from utils.translations import Translation
from utils.labels import Label
from datetime import datetime
#import pandas as pd

# Get execution time
start = datetime.now()

# Import german labeled ad-hoc and their english counterpart and goldstandards
imp = Import()
data = imp.findcounterpart()
goldstd = imp.importgold()

# Extract subdf of the 15 random selected documents, which has also
# been manually labelled
#random_data = pd.read_csv('data/random_docs.csv')
#test_data = data[data['hash_y'].isin(random_data['hash_y'])].reset_index(drop=True)

# Compute (cosine) similarity between goldstandards and english sentences
trans = Translation()
cosine_scores = trans.cosine_similarity(data, goldstd)

# Compute the english labeled dataframe basing on the cosine similarity outcome
label = Label()
english_labeled = label.merge_labels(cosine_scores, goldstd)

cosine_scores.to_pickle('data/cosine_scores.pkl')
english_labeled.to_pickle('data/english_goldstandards.pkl')

print(datetime.now() - start)
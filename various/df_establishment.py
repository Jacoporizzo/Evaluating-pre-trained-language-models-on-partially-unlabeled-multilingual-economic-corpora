'''
Data preprocessing steps in order to set up the final 
df to use. This script has been executed using a GPU
and the results stored locally as pickle files. Adjust 
the path of the data location before running the script. 
'''

from utils.imports import Import
from utils.translations import Translation
from utils.labels import Label
from datetime import datetime

# Get execution time
start = datetime.now()

# Import german labeled ad-hoc and their english counterpart and goldstandards
imp = Import()
data = imp.findcounterpart()
goldstandards = imp.importgold()

# Compute (cosine) similarity between goldstandards and english sentences
trans = Translation()
cosine_scores = trans.cosine_similarity(data, goldstandards)

# Compute the english labeled dataframe basing on the cosine similarity outcome
label = Label()
english_labelled = label.merge_labels(cosine_scores, goldstandards)

# Save dfs in data folder
cosine_scores.to_pickle('data/cosine_scores.pkl')
english_labelled.to_pickle('data/english_goldstandards.pkl')

# Print total computation time
print(datetime.now() - start)
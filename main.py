# First way of setting up the df 
# import os
# os.chdir('C:/Users/Jacopo/Desktop/MA')
from utils.imports import Import
from utils.translations import Translation
from utils.labels import Label
from datetime import datetime

# Get execution time
start = datetime.now()

# Import german labeled ad-hoc and their english counterpart 
imp = Import()
data = imp.findcounterpart()

# Compute (cosine) similarity between translated sentences
trans = Translation()
cosine_scores = trans.cosine_similarity(data[0:10])

# Import the goldstandards
goldstd = imp.importgold()

# Compute the english labeled dataframe basing on the cosine similarity outcome
label = Label()
english_labeled = label.merge_labels(goldstd, cosine_scores)

print(datetime.now() - start)
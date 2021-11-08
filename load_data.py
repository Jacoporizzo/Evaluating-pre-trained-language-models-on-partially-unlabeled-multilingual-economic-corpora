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

# Extract split doc variables for testing
english_docs = data['bodyText_x'][0:10]
german_docs = data['bodyText_y'][0:10]

# Compute (cosine) similarity between translated sentences
trans = Translation()
cosine_scores = trans.cosine_similarity(english_docs, german_docs)

# Import the goldstandards and pick only the present sentences
goldstd = imp.importgold()

# Compute the english labeled dataframe basing on the cosine similarity outcome
label = Label()
english_labeled = label.merge_labels(data, goldstd, cosine_scores)

print(datetime.now() - start)

# On my laptop about 180 min for entire dataset
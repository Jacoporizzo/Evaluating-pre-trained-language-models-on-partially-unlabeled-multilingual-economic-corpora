'''
Data preprocessing steps in order to set up the final 
df to use. This script has been executed using a GPU
and the results stored locally as pickle files. Adjust 
the path of the data location before running the script. 
'''

from utils.imports import Import
from utils.translations import Translation
from utils.labels import Label
from utils.helpers import Helper
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

# Get df to use for finetuning
helper = Helper()
df_finetune = helper.get_inputs(english_labelled)
df_finetune.to_pickle('data/df_finetune.pkl')

# Create df with document-level labelled instances
df_fulltext = helper.fulltext_labels(english_labelled)
df_fulltext.to_pickle('data/labelled_doc.pkl')

# Create df with 8k items linked to german classes
forms8k = imp.import8k()
linked_data = helper.link_classes(forms8k)
linked_data.to_pickle('data/linked_8k.pkl')

# Create df for 8k with only text and german class
text8k_class = helper.get_inputs_8k(linked_data)
text8k_class.to_pickle('data/8k_text_gerclass.pkl')

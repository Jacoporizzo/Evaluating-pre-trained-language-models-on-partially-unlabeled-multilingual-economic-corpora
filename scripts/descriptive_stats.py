'''
Descriptive statistics for the orginal data.
'''

from utils.imports import Import

# Import data
imports = Import()
gold_data = imports.importgold()

# Bar chart of the labels (absolute frequency).
# The amount and distribution of the labels is 
# the same for the goldstandards and the SBERT's output.
labels = gold_data.iloc[:,4:26]
bars = labels.sum(axis = 0)
bc = bars.plot.bar()
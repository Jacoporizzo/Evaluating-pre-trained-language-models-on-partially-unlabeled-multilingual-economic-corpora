'''
Descriptive statistics for the orginal data and
in general for base datasets used throughout
the work.
'''
import pickle
import pandas as pd
import plotnine as pn
from utils.imports import Import

###########################################################
### BASE goldstandards dataset
imp = Import()
gold_data = imp.importgold()

# Bar chart absolute and relative frequency for 
# ground (i.e. true) labels
labels = gold_data.iloc[:,4:26]
bars = labels.sum(axis = 0)

category =  ['Earnings',
             'SEO',
             'Management',
             'Guidance',
             'Gewinnwarnung',
             'Beteiligung',
             'Dividende',
             'Restructuring',
             'Debt',
             'Law',
             'Großauftrag',
             'Squeeze',
             'Insolvenzantrag',
             'Insolvenzplan',
             'Delay',
             'Split',
             'Pharma_Good',
             'Rückkauf',
             'Real_Invest',
             'Delisting',
             'Irrelevant',
             'Empty']

bars_df = bars.reset_index().rename(columns = {0: 'total'})
(pn.ggplot(bars_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Absolute frequency', x = '', title = "German goldstandards' true labels")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

bars_df['proportion'] = bars_df['total']/sum(bars_df['total'])
bars_df['index'] = pd.Categorical(bars_df['index'], categories = category)
(pn.ggplot(bars_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative frequency', x = '', title = "Classes distribution")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

###########################################################
### BASE English goldstandards dataset

# Of course the absolute and realtive frequency for
# each label are equal to the one above from the base 
# goldtsandards dataset. Though this is only for reason
# of providing a complete overview

# Import data
english_goldstandards = pickle.load(open('data/english_goldstandards.pkl', 'rb'))

# Bar charts absolute and relative freq.
english_labels = english_goldstandards.iloc[:,6:28]
english_bars = english_labels.sum(axis = 0)

english_bars_df = english_bars.reset_index().rename(columns = {0: 'total'})
(pn.ggplot(english_bars_df, pn.aes(x = 'index', y = 'total'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Absolute freqeuncy', x = '', title = "English goldstandards' true labels")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))

english_bars_df['proportion'] = english_bars_df['total']/sum(english_bars_df['total'])
(pn.ggplot(english_bars_df, pn.aes(x = 'index', y = 'proportion'))
     + pn.geom_col(color = 'blue', fill = 'blue')
     + pn.labs(y = 'Relative freqeuncy', x = '', title = "English goldstandards' true labels")
     + pn.theme(axis_text_x = pn.element_text(angle = 90)))
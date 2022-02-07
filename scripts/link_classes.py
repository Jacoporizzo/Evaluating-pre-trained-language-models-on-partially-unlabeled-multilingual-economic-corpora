from utils.imports import Import
import pandas as pd
import numpy as np
from utils.helpers import Helper

imp = Import()
data = imp.import8k()

# Remove all items that cannot be linked to a german class
items_to_remove = [np.float64(1.02),
                   np.float64(1.04),
                   np.float64(2.04),
                   np.float64(2.06),
                   np.float64(3.02),
                   np.float64(3.03),
                   np.float64(4.01),
                   np.float64(4.02),
                   np.float64(5.04),
                   np.float64(5.05),
                   np.float64(5.06),
                   np.float64(5.07),
                   np.float64(6.01),
                   np.float64(6.02),
                   np.float64(6.03),
                   np.float64(6.04),
                   np.float64(6.05),
                   np.float64(9.01)]

final_df = data[~data['ItemNumber'].isin(items_to_remove)].reset_index(drop = True)

helper = Helper()
labels_names = helper.get_labels_names()
link_classes = {np.float64(1.01): ['SEO', 'Debt'],
                np.float64(1.03): ['Insolvenzplan', 'Insolvenzantrag'],
                np.float64(2.01): ['Beteiligung', 'Real_Invest'],
                np.float64(2.02): ['Guidance', 'Gewinnwarnung'],
                np.float64(2.03): 'Debt',
                np.float64(2.05): 'Restructuring',
                np.float64(3.01): 'Delisting',
                np.float64(5.01): 'Management',
                np.float64(5.02): 'Management',
                np.float64(5.03): 'Split',
                np.float64(5.08): 'Management',
                np.float64(7.01): 'Dividende',
                np.float64(8.01): 'Rückkauf'}

link_classes = {1.01: ['SEO', 'Debt'],
                1.03: ['Insolvenzplan', 'Insolvenzantrag'],
                2.01: ['Beteiligung', 'Real_Invest'],
                2.02: ['Guidance', 'Gewinnwarnung'],
                2.03: 'Debt',
                2.05: 'Restructuring',
                3.01: 'Delisting',
                5.01: 'Management',
                5.02: 'Management',
                5.03: 'Split',
                5.08: 'Management',
                7.01: 'Dividende',
                8.01: 'Rückkauf'}

link_classes = {'SEO': np.float64(1.01),
                'Debt': np.float64(1.01),
                'Insolvenzplan': np.float64(1.03),
                'Insolvenzantrag': np.float64(1.03),
                'Beteiligung': np.float64(2.01),
                'Real_Invest': np.float64(2.01),
                'Guidance': np.float64(2.02),
                'Gewinnwarnung': np.float64(2.02),
                'Debt': np.float64(2.03),
                'Restructuring': np.float64(2.05),
                'Delisting': np.float64(3.01),
                'Management': np.float64(5.01),
                'Management': np.float64(5.02),
                'Split': np.float64(5.03),
                'Management': np.float64(5.08),
                'Dividende': np.float64(7.01),
                'Rückkauf': np.float64(8.01)}

items = [np.float64(1.01),
         np.float64(1.03),
         np.float64(2.01),
         np.float64(2.02),
         np.float64(2.03),
         np.float64(2.05),
         np.float64(3.01),
         np.float64(5.01),
         np.float64(5.02),
         np.float64(5.03),
         np.float64(5.08),
         np.float64(7.01),
         np.float64(8.01)]

test_df = final_df[0:10]

for var in labels_names:
    key = [k for k, v in link_classes.values() if v == var]
    test_df[var] = [True for i in test_df['ItemNumber'] if i == key]
    
# This works apart for Grossauftrag and variable appending
for var in labels_names[0:5]:
    class_var = []
    for item in items:
        if var in link_classes.get(item):
            for item_nr in test_df['ItemNumber']:
                if item == item_nr:
                    class_var.append('True')
                else:
                    class_var.append('False')
        else:
             pass
    test_df[var] = class_var 
        

for var in labels_names:
    class_var = []
    for item in items:
        for item_nr in test_df['ItemNumber']:
            if item == item_nr and var in link_classes.get(item):
                class_var.append(True)
            else:
                class_var.append(False)
    
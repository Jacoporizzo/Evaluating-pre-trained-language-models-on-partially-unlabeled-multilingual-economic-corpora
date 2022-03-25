'''
Topic: Master thesis
Description: Class that creates function that help to get
             the datasets for making the plots. 

Created on: 24 March 2022
Created by: Jacopo Rizzo
'''
import pandas as pd
import numpy as np
from utils.helpers import Helper

class PlotData:

    helper = Helper()

    def labels_doc(self, items_cls, prediction = False):
        '''
        Get document labels as integers.

        Parameters
        ----------
        items_cls : df
            Df containing the grouped document labels, true and predicted.
            Output of DocLevel.remove_empty_class().
        prediction : bool
            Whether to get the predicted or true labels. Default to False.

        Returns
        -------
        doc_labs : list
            List containing integers of true or predicted labels.
        '''
        doc_labs = []
        for j in range(len(items_cls)):
            if prediction:
                arr = np.nonzero(items_cls['label_predicted'][j])
            else:
                arr = np.nonzero(items_cls['label_true'][j])
            multi = []
            for i in range(len(arr[0])):
                multi.append(arr[0][i])
            doc_labs.append(multi)

        return doc_labs

    def labels_names(self, labels):
        '''
        Get the labels names for input.

        Parameters
        ----------
        labels : list
            List of labels (integers). Output of helper.predicted_labels(),
            or of labels_doc(...,prediction = True) for document predictions.

        Returns
        -------
        lab_names : list
            List of labels names.
        '''
        lab_names = self.helper.get_labels_names()
        dicts = dict(zip(range(0,22), lab_names))

        lab_names = []
        for i in labels:
            if len(i) == 0:
                lab_names.append(['NC'])
            elif len(i) == 1:
                for j in i:
                    lab_names.append([dicts[j]])
            else:
                multi = []
                for t in i:
                    multi.append(dicts[t])
                lab_names.append(multi)

        return lab_names

    def labels_complete_df(self, labels):
        '''
        Create dataframe for input labels.

        Parameters
        ----------
        labels: list
            List with predicted or true labels names, i.e. output of labels_names().

        Returns
        -------
        labs_df : df
            Pandas df with absolute and relative frequency of input labels.
        '''
        labs = [item for items in labels for item in items]
        labs_df = pd.DataFrame.from_dict(dict([[x, labs.count(x)] for x in set(labs)]), orient = 'index').rename(columns = {0 : 'total'})
        labs_df['proportion'] = labs_df['total']/sum(labs_df['total'])
        
        return labs_df.reset_index().rename(columns={'index': 'label'})
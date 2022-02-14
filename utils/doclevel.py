'''
Topic: Master thesis
Description: Class for documentlevel analysis

Created on: 14 February 2022
Created by: Jacopo Rizzo
'''
import pandas as pd
import pickle
import numpy as np
from utils.helpers import Helper

class DocLevel:

    helper = Helper()

    def doc_labels(self, data):
        '''
        Aggregate true labels on document level.

        Parameters
        ----------
        data : df
            Dataframe of the test data.

        Returns
        -------
        hashed_true : df
            Dataframe containing aggregated true labels per hash (i.e. document).

        '''
        hashs, labs = [], []
        unique_hash = data['hash'].unique()
        
        for uhash in unique_hash:
            subset = data[data['hash'] == uhash]
            aggregate = np.array(list(map(any, zip(*subset['label']))), dtype = int)
            hashs.append(uhash)
            labs.extend([aggregate])

        hashed_true = pd.DataFrame({'hash': hashs,
                                    'label': [l.tolist() for l in labs]})
            
        return hashed_true

    def doc_predictions(self, data, predictions, threshold = 0.5):
        '''
        Aggregate true labels on document level.

        Parameters
        ----------
        data : df
            Dataframe of the test data.
        predictions: df
            List of dictionaries of the score for each label of the
            predictions (output of the model).
        threshold: float, optional
            Threshold to set for automatic classification in one class. Must be
            in the range [0,1]. The default is 0.5
        
        Returns
        -------
        hashed_true : df
            Dataframe containing aggregated predicted labels per hash (i.e. document).

        '''
        df_thresh = self.helper.threshold_classification(predictions, threshold = threshold)
        preds = self.helper.predicted_labels(df_thresh)

        doc_preds = pd.DataFrame({'hash': data['hash'],
                                  'label': preds})

        zero = np.zeros(22)
        binarized = []        
        for i in doc_preds['label']:
            zero = np.zeros(22, dtype = int)
            zero[i] = 1
            binarized.append(zero.tolist())

        doc_preds['label'] = binarized 

        hashs, labs = [], []
        unique_hash = doc_preds['hash'].unique()

        for uhash in unique_hash:
            subset = doc_preds[doc_preds['hash'] == uhash]
            aggregate = np.array(list(map(any, zip(*subset['label']))), dtype = int)
            hashs.append(uhash)
            labs.extend([aggregate])
            
        hashed_preds = pd.DataFrame({'hash': hashs, 
                                     'label': [l.tolist() for l in labs]})
        
        return hashed_preds
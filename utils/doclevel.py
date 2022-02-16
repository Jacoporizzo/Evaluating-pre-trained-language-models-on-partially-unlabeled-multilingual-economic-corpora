'''
Topic: Master thesis
Description: Class for document-level analysis

Created on: 14 February 2022
Created by: Jacopo Rizzo
'''
import pandas as pd
import pickle
import numpy as np
from utils.helpers import Helper
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
        df = pd.DataFrame(data)
        hashs, labs = [], []
        unique_hash = df['hash'].unique()
        
        for uhash in unique_hash:
            subset = df[df['hash'] == uhash]
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
        df = pd.DataFrame(data)
        df_thresh = self.helper.threshold_classification(predictions, threshold = threshold)
        preds = self.helper.predicted_labels(df_thresh)

        doc_preds = pd.DataFrame({'hash': df['hash'],
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

    def remove_empty_class(self, true_labels, predicted_labels):
        '''
        Remove truth and prediction for the class "Empty" and consequently
        all those rows (i.e. documents) which are labelled only in this class

        Parameters
        ----------
        true_labels : dataframe
            Pandas df containing binarized true labels for each hash.
        predicted_labels : dataframe
            Pandas df containing binarized predicted labels for each hash.

        Returns
        -------
        df : dataframe
            Pandas df with binarized truth and prediction.

        '''
        data = pd.merge(true_labels, predicted_labels, on = 'hash', how = 'inner', suffixes = ('_true', '_predicted'))
        df = data.copy()
        
        # Remove last element from both labels vars, since this is fot the empty class
        for i in df['label_true']:
            i.pop(21)
            
        for j in df['label_predicted']:
            j.pop(21)
        
        to_remove = []
        for l in range(len(df)):
            if ((all(t == 0 for t in df['label_true'][l])) or (all(p == 0 for p in df['label_predicted']))):
                to_remove.append(True)
            else:
                to_remove.append(False)
        
        df['to_remove'] = to_remove
        df.drop(df[df['to_remove'] == True].index, inplace = True)
        
        return df

    def doc_evaluations(self, true_labels, predicted_labels, level = 'global', average = 'macro'):
        '''
        Compute evaluation metrics on dcoument level

        Parameters
        ----------
        true_labels : list
            List with true labels for each hash. Can be established using 
            helper.actual_labels().
        predicted_labels : list
            List with predicted labels for each hash. Can be established using 
            helper.actual_labels().
        level : str, optional
            Level on which to perform evaluation. Either local or global.
            The default is 'global'.
        average : str, optional
            Method with which to compute the local evalution. 
            The default is 'macro'.

        Raises
        ------
        ValueError
            Provided invalid argument.

        Returns
        -------
        metrics
            DIctionary (global level) or df (local level) with the 
            evaluation metrics.

        '''
        # Control input parameters
        VALID_EVAL = {'macro', 'micro', 'weighted', None, 'samples'}
        VALID_LEVELS = {'global', 'local'}
        if level not in VALID_LEVELS:
            raise ValueError("Level must be one of %r." % VALID_LEVELS)

        if average not in VALID_EVAL:
            raise ValueError("Average must be one of %r." % VALID_EVAL)

        # Binarize as sparse matrix truth and predictions
        bin_true = np.array(list(true_labels))
        bin_preds = np.array(list(predicted_labels))

        # Compute metrics
        acc = accuracy_score(bin_true, bin_preds)

        if level == 'global':
            pre, rec, f1, _ = precision_recall_fscore_support(bin_true, bin_preds, average = average)
            metrics = {'accuracy': acc,
                       'f1': f1,
                       'precision': pre,
                       'recall': rec}
        else:
            labels_names = self.helper.get_labels_names()
            labels_names.remove('Empty')
            pre, rec, f1, sup = precision_recall_fscore_support(bin_true, bin_preds)
            metrics = pd.DataFrame({'labels': labels_names,
                                    'precision': pre,
                                    'recall': rec,
                                    'f1': f1, 
                                    'support': sup})

        return metrics
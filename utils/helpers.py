'''
Topic: Master thesis
Description: Class containing helper functions

Created on: 12 January 2022
Created by: Jacopo Rizzo
'''
import pandas as pd
import pickle
import torch
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, 
                             accuracy_score, 
                             confusion_matrix)

class Helper:

    def get_labels_names(self, data = None):
        '''
        Get the names of the labels.

        Parameters
        ----------
        data : Dataframe
            Provided dataframe with labels.

        Returns
        -------
        labels_names : list
            Ordered list with the names of the different labels.

        '''
        data = pickle.load(open('data/english_goldstandards.pkl', 'rb'))

        # Columns subsetting if using provided dataframe
        labels = list(data.columns)
        labels_names = labels[6:28]

        return labels_names

    def get_inputs(self, data):
        '''
        Set up the dataframe to use for fine-tuning.

        Parameters
        ----------
        data : Dataframe
            Provided dataframe.

        Returns
        -------
        df : Dataframe
            Dataframe to use for fine-tuning, containing list of labels and text.

        '''
        # Columns subsetting if using provided dataframe
        binary_labels = data.iloc[:,6:28].astype(int)

        df = pd.DataFrame()
        df['text'] = data['English_sentences']
        df['label'] = binary_labels.values.tolist()

        return df

    def fulltext_labels(self, data, goldstandards):
        '''
        Create dataframe containing the entire text of a documents with its labels.

        Parameters
        ----------
        data : dataframe
            Dataframe containing the entire documents' texts. Output from
            Import.findcounterpart()
        goldstandards : datframe
            Dataframe containing the Englich goldstandards. Already saved and
            can be directly imported. 

        Returns
        -------
        df : dataframe
            Dataframe with aggregated labels for each document.

        '''
        hash_list = list(goldstandards['English_hash'].unique())

        # Create dataframe for texts' labels
        df = pd.DataFrame(columns = {'hash', 'text', 'labels'})

        for i in range(len(hash_list)):
            labels = goldstandards[goldstandards['English_hash'] == hash_list[i]].iloc[:, 6:28].astype(int)
            binary_labels = labels.values.tolist()
            aggregate = np.array(list(map(any, zip(*binary_labels))), dtype = int)
            
            # Get entire text for given document
            idx = data.index[data['hash_x'] == hash_list[i]][0]
            
            text = data['bodyTextRaw_x'][idx]
            title = data['titleText_x'][idx]

            if title not in text:
                full_text = title + '. ' + text
            else:
                full_text = text

            df.loc[i] = [hash_list[i], full_text, list(aggregate)]

        return df

    def actual_labels(self, data):
        '''
        Get the true labels from the data.

        Parameters
        ----------
        data : Dataset
            Raw (test) data with labels (type arrow_dataset.Dataset).

        Returns
        -------
        labels_nr : list
            List of lists with label(s) of input data.

        '''
        labels = []

        for tensor in data:
            labels.append([idx for idx in range(len(tensor)) if tensor[idx] == 1])
            #labels.append(torch.where(tensor == 1)[0])

        #labels_nr = [item.tolist() for item in labels]

        return labels

    def predicted_labels_scores(self, actual, predicted):
        '''
        Get the predicted labels by the model and their corrseponding
        prediction score.

        Parameters
        ----------
        actual : list
            List of true labels (output of actual_labels).
        predicted : list
            List of dictionaries with predictions (output of pipeline process).

        Returns
        -------
        predicted_labels : List
            List of dictionaries with predictions and corresponding scores.

        '''
        predicted_labels = []
        for i in range(len(actual)):
            n_labels = len(actual[i])
            sort_pred = sorted(predicted[i], key = lambda x: x['score'], reverse = True)
            predicted_labels.append(sort_pred[0:n_labels])

        return predicted_labels

    def predicted_labels(self, prediction):
        '''
        Get list of predicted labels.

        Parameters
        ----------
        prediction : list
            List of dictinaries (output of predicted_labels_score).

        Returns
        -------
        pred_labs : list
            List of predicted label(s) for given text.

        '''
        labels = [[d["label"] for d in row] for row in prediction]

        pred_labs = []
        for l in range(len(labels)):
            nr_lab = len(labels[l])
            multi_labels = []
            for pos in range(nr_lab):
                multi_labels.append(int(labels[l][pos].split('LABEL_',1)[1]))
                multi_labels.sort()
            pred_labs.append(multi_labels)
        
        return pred_labs
            
    def evaluation_scores(self, true_labels, predicted_labels, level = 'global', average = 'macro'):
        '''
        Compute the evaluation metrics (accuracy only globally) for the test dataset.

        Parameters
        ----------
        true_labels : list
            List of lists with the true labels (output of actual_labels).
        predicted_labels : list
            List of lists with the predicted labels (output of predicted_labels).
        level : str, optional
            Whether computing the metrics globally ('global') over the entire datset
            or on a class basis ('local'). If global, then the output for precision
            recall and f1 is the mean of the outputs for each single class.
            The default is 'global'.
        average : str, optional
            Art of average for global scores for the entire dataset. Only valid
            for 'global'. For 'local' automatically set to None. The default is 'macro'.

        Returns
        -------
        metrics
            Dictionary (for global level) or dataframe (for local level) 
            with the the four evaluation metrics.

        '''
        VALID_EVAL = {'macro', 'micro', 'weighted', None, 'samples'}
        VALID_LEVELS = {'global', 'local'}
        if level not in VALID_LEVELS:
            raise ValueError("Level must be one of %r." % VALID_LEVELS)

        if average not in VALID_EVAL:
            raise ValueError("Average must be one of %r." % VALID_EVAL)

        true = [item for sublist in true_labels for item in sublist]
        pred = [item for sublist in predicted_labels for item in sublist]

        acc = accuracy_score(true, pred)

        if level == 'global':
            precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average = average)
            metrics = {'accuracy': acc,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall}
        else:
            lab_names = self.get_labels_names()
            precision, recall, f1, _ = precision_recall_fscore_support(true, pred)
            metrics = pd.DataFrame({'labels': lab_names,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1})

        return metrics

    def eval_confusion(self, true_labels, predictions):
        '''
        Create confusion matrix for predicted and actual labels.

        Parameters
        ----------
        true_labels : list
            List of true labels. Output of actual_labels().
        predictions : list
            List of predicted labels. Output of predicted_labels().

        Returns
        -------
        cm : dataframe
            Confusion matrix for all labels as dataframe.

        '''
        true = [item for sublist in true_labels for item in sublist]
        prediction = [item for sublist in predictions for item in sublist]
        
        lab_names = Helper().get_labels_names()
        dicts = dict(zip(range(0,22), lab_names))
        
        cm = pd.DataFrame(confusion_matrix(true, prediction)).rename(columns = dicts, index = dicts)
        return cm

    def link_classes(self, forms8k):

        labels_names = self.get_labels_names()

        # Items to be removed
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

        # Remove items form data
        data = forms8k[~forms8k['ItemNumber'].isin(items_to_remove)].reset_index(drop = True)

        # Craete dictionary for linking 
        link_classes = {np.float64(1.01): ['SEO', 'Debt'],
                        np.float64(1.03): ['Insolvenzplan', 'Insolvenzantrag'],
                        np.float64(2.01): ['Beteiligung', 'Real_Invest'],
                        np.float64(2.02): ['Guidance', 'Gewinnwarnung'],
                        np.float64(2.03): ['Debt'],
                        np.float64(2.05): ['Restructuring'],
                        np.float64(3.01): ['Delisting'],
                        np.float64(5.01): ['Management'],
                        np.float64(5.02): ['Management'],
                        np.float64(5.03): ['Split'],
                        np.float64(5.08): ['Management'],
                        np.float64(7.01): ['Dividende'],
                        np.float64(8.01): ['Rückkauf']}

        data['label_string'] = data['ItemNumber'].map(link_classes)

        for var in labels_names:
            class_var = []
            for item in data['label_string']:
                if var in item:
                    class_var.append(True)
                else:
                    class_var.append(False)
            data[var] = class_var

        data.drop('label_string', axis = 1, inplace = True)

        return data
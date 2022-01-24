'''
Topic: Master thesis
Description: Class containing helper functions

Created on: 12 January 2022
Created by: Jacopo Rizzo
'''
import pandas as pd
import pickle
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

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

    def predicted_labels_score(self, actual, predicted):
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
            
    def evaluation_scores(self, true_labels, predicted_labels, eval_schema = 'macro'):
        '''
        Compute the evaluation metrics for the test dataset.

        Parameters
        ----------
        true_labels : list
            List of lists with the true labels (output of actual_labels).
        predicted_labels : list
            List of lists with the predicted labels (output of predicted_labels).
        eval_schema : string, optional
            Schema to use for evaluation see doc of sklearn-metrics for a 
            complete list of available schemes. The default is 'micro'.

        Returns
        -------
        dict
            Dictionary with the the four evaluation metrics.

        '''
        true = [item for sublist in true_labels for item in sublist]
        pred = [item for sublist in predicted_labels for item in sublist]

        acc = accuracy_score(true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average = eval_schema)

        return {'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall}

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
            Confusiion matrix for as dataframe.

        '''
        true = [item for sublist in true_labels for item in sublist]
        prediction = [item for sublist in predictions for item in sublist]
        
        lab_names = Helper().get_labels_names()
        dicts = dict(zip(range(0,22), lab_names))
        
        cm = pd.DataFrame(confusion_matrix(true, prediction)).rename(columns = dicts, index = dicts)
        return cm
'''
Topic: Master thesis
Description: Class containing helper functions

Created on: 12 January 2022
Created by: Jacopo Rizzo
'''
import pandas as pd

class Helper:

    def get_labels_names(self, data):
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
        df['Text'] = data['English_sentences']
        df['Labels'] = binary_labels.values.tolist()

        return df

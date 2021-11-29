'''
Topic: Master thesis
Description: Class to evaluate the output 

Created on: 29 November 2021 
Created by: Jacopo Rizzo
'''
import pandas as pd
import numpy as np 

class Evaluation:

    def is_equal(self, manual_labels):
        '''
        Compares equality between SBERT's output and (manual) goldstandard for
        a given entry.

        Parameters
        ----------
        manual_labels : dataframe
            Dataframe of manual translated goldstandards.

        Returns
        -------
        df : dataframe
            Dataframe with equlity check as boolena variable.

        '''
        manual = manual_labels['English_manual']
        output = manual_labels['English_cosine_output']

        manual_stripped = manual.str.replace('[^\w\s]', '').str.lower().str.replace(' ', '')
        out_stripped = output.str.replace('[^\w\s]', '').str.lower().str.replace(' ', '')
        
        # Compare SBERT's output with manual translated version as boolen value
        comparison = (manual_stripped == out_stripped)

        df = pd.DataFrame([manual, output, comparison, manual_stripped, out_stripped]).T
        df.columns = ['English_manual','English_cosine_output','Is_equal','Manual_stripped', 'Cosine_output_stripped'] 
        
        return df

    def create_confusion(self, equal_comparison):
        '''
        Create confusion matrix for test dataset.

        Parameters
        ----------
        equal_comparison : dataframe
            Dataframe derived with is_equal().

        Returns
        -------
        confusion_matrix : dataframe
            Confusion matrix for input.

        '''
        model_output, actual_goldstandard = [], []

        # Get needed data
        is_equal = equal_comparison['Is_equal']
        manual_stripped = equal_comparison['Manual_stripped']
        cosine_stripped = equal_comparison['Cosine_output_stripped']

        # Loop over data and create lists for confusion
        for value in is_equal.index:
            if is_equal[value] == True:
                model_output.append(1)
                actual_goldstandard.append(1)
            else:
                if manual_stripped[value] in cosine_stripped[value]:
                    model_output.append(1)
                    actual_goldstandard.append(0)
                elif cosine_stripped[value] in manual_stripped[value]:
                    model_output.append(0)
                    actual_goldstandard.append(1)
                else:
                    model_output.append('NA')
                    actual_goldstandard.append('NA')

        df_conf = pd.DataFrame([model_output, actual_goldstandard]).T
        df_conf.columns = ['Model_output', 'Actual_goldstandard']

        # Create confusion matrix
        confusion_matrix = pd.crosstab(df_conf['Actual_goldstandard'],
                                       df_conf['Model_output'],
                                       rownames = ['Actual'], colnames = ['Predicted'])
        
        # Shape of output matrix is the following
        #        Prediction (SBERT's output)
        #          |    0         1
        # ---------|-----------------
        #        0 |   TN         FP 
        # Actual   |
        #   GS   1 |   FN         TP 
        #          |

        # Rename columns and rows for better overview
        confusion_matrix.columns = ['Predicted_0', 'Predicted_1', 'NA']
        confusion_matrix.index = ['Actual_0', 'Actual_1', 'NA']

        return confusion_matrix
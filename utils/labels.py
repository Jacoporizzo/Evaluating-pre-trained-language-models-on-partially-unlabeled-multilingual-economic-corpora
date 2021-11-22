'''
Topic: Master thesis
Description: Class for labeling the english sentences

Created on: 29 October 2021 by Jacopo Rizzo
Last modified on: 29 October 2021
'''
import pandas as pd
import numpy as np

class Label:

    def merge_labels(self, goldstandards, cosine_similarity):
        '''
        Merge the result of the cosine-similarity computation with the 
        goldstandards in order to label the english sentences. Works only for 
        german sentences in goldstandard, which are labeled singularly.

        Parameters
        ----------
        goldstandards : Dataframe
            Dataframe of the goldstandards, i.e. Import.importgold.
        
        cosine_similarity : Dataframe
            Dataframe of the highest cosine similarity between two sentences, i.e.
            result of Translation.cosine_similarity.

        Returns
        -------
        Dataframe
            English sentences with their labels, ready to use for the classification.

        '''
        cosines = cosine_similarity.copy()
        cosines.rename(columns = {'German_sentences': 'Sentences'}, inplace = True)
        goldstd = goldstandards.reset_index(drop = True)

        hashs_lst = cosines['German_hash'].unique()
        
        labeled_sents = []
        for has in hashs_lst:
            subset_gold = goldstd[goldstd['Hashs'] == has].reset_index()
            subset_cosine = cosines[cosines['German_hash'] == has].reset_index()

            # Eliminate punctuation, beginning whitespaces and set all to lowercase
            gold_plain = subset_gold['Sentences'].str.replace('[^\w\s]', '').str.lower().str.lstrip()
            cosine_plain = subset_cosine['Sentences'].str.replace('[^\w\s]', '').str.lower().str.lstrip()

            for sentence in cosine_plain:
                if (gold_plain.str.contains(sentence, regex = False)).any():
                    idx = np.where(gold_plain.str.contains(sentence, regex = False))[0][0]
                    labeled_sents.append(subset_gold['Sentences'][idx])
                else:
                    labeled_sents.append('Not a goldstandard')

        cosines['Sentences'] = pd.Series(labeled_sents)

        cosines = cosines.drop(cosines[cosines['Sentences'] == 'Not a goldstandard'].index).reset_index(drop = True)

        # Concatenate strings together
        grouped_cosine = pd.DataFrame(cosines.groupby('Sentences'))

        eng_sent, cos_mean, ger_hash, eng_hash = [], [], [], []

        grouped_sents = grouped_cosine[1]
        
        for group in grouped_sents:
            eng_sent.append(group['English_sentences'].str.cat(sep = ' '))
            cos_mean.append(group['Cosine_score'].mean())
            ger_hash.append(group['German_hash'].unique()[0])
            eng_hash.append(group['English_hash'].unique()[0])

        # Create dataframe to merge with goldstandards
        df_to_merge = pd.DataFrame()
        df_to_merge['Sentences'] = grouped_cosine[0]
        df_to_merge['English_sentences'] = pd.Series(eng_sent)
        df_to_merge['Cosine_score'] = pd.Series(cos_mean)
        df_to_merge['English_hash'] = pd.Series(eng_hash)
        df_to_merge['German_hash'] = pd.Series(ger_hash)
        
        # Merge translations with goldstandards
        df_merge = df_to_merge.merge(goldstd, how = 'inner', on = 'Sentences')
        df_merge = df_merge.sort_values(['Hashs', 'SentenceNr']).reset_index(drop = True)
        df_merge = df_merge.drop(['Hashs'], axis = 1)

        return df_merge
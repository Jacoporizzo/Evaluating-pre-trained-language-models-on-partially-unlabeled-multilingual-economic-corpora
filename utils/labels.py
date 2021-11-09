'''
Topic: Master thesis
Description: Class for labeling the english sentences

Created on: 29 October 2021 by Jacopo Rizzo
Last modified on: 29 October 2021
'''
import pandas as pd
import numpy as np

class Label:

    def merge_labels(self, counterparts, goldstandards, cosine_similarity):
        '''
        Merge the result of the cosine-similarity computation with the 
        goldstandards in order to label the english sentences. Works only for 
        german sentences in goldstandard, which are labeled singularly.

        Parameters
        ----------
        counterparts : Dataframe
            Dataframe of the base data, i.e. output of Import.findcounterpart.

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
        cosine_similarity.rename(columns = {'German_sentences': 'Sentences'}, inplace = True)
        goldstd = goldstandards.reset_index(drop = True)

        indices = []
        for sentence in cosine_similarity['Sentences']:
            if (goldstd['Sentences'].str.contains(sentence, regex = False)).any():
                idx = np.where(goldstd['Sentences'].str.contains(sentence, regex = False))[0][0]
                indices.append(goldstd['Sentences'][idx])
            else:
                indices.append('Not a goldstandard')

        cosine_similarity['Sentences'] = pd.Series(indices)

        df_merge = cosine_similarity.merge(goldstd, how = 'inner', on = 'Sentences')

        english_hashs = []
        for hashs in df_merge['Hashs']:
            english_hashs.append(counterparts[counterparts['hash_y'] == hashs]['hash_x'].values)

        hashs_series = pd.Series([item for hashs in english_hashs for item in hashs])

        df_merge['Hashs'] = hashs_series
        df_merge = df_merge.drop(['Sentences', 'Cosine_score'], axis = 1)
         
        return df_merge
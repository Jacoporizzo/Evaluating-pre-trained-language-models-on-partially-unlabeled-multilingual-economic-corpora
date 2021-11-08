'''
Topic: Master thesis
Description: Class for labeling the english sentences

Created on: 29 October 2021 by Jacopo Rizzo
Last modified on: 29 October 2021
'''
import pandas as pd

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
        df_merge = cosine_similarity.merge(goldstandards, how = 'inner', on = 'Sentences')

        english_hashs = []
        for hashs in df_merge['Hashs']:
            english_hashs.append(counterparts[counterparts['hash_y'] == hashs]['hash_x'].values)

        hashs_series = pd.Series([item for hashs in english_hashs for item in hashs])

        df_merge['Hashs'] = hashs_series
        df_merge = df_merge.drop(['Sentences', 'Cosine_score'], axis = 1)
         
        return df_merge

        '''
        gold_sentences = goldstandards['Sentences']
        single_sentences = cosine_similarity['German_sentences']
        indices = []

        for sentence in single_sentences:
            indices.append(np.where(gold_sentences.str.contains(sentence, regex = False))[0])

        fill_empty_arrays = []
        for idx in indices:
            if idx.size == 1:
                fill_empty_arrays.append(int(idx))
            else:
                fill_empty_arrays.append(float(np.array(0.1)))

        sentence_from_idx = []
        for idx in fill_empty_arrays:
            try:
                sentence_from_idx.append(gold_sentences[idx])
            except:
                sentence_from_idx.append('Empty')

        cosine_similarity['German_sentences'] = pd.Series(sentence_from_idx)

        return cosine_similarity
        '''
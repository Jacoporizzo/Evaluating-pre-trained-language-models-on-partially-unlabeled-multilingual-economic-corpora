'''
Topic: Master thesis
Description: Class for computing the similarity between translated sentences

Created on: 29 October 2021 by Jacopo Rizzo
Last modified on: 29 October 2021
'''
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from utils.imports import Import

class Translation:

    def concatenation(self, languages):
        '''
        Concatenate together in one list the single german and english 
        sentences of the same translated document.

        Parameters
        ----------
        languages : list
            List of lists, each containing the split english and german 
            sentences of the same document.

        Returns
        -------
        concat : list
            List of concatenated lists of input languages.

        '''
        english = languages[0]
        german = languages[1]

        # Concatenate the english and german version of the documents, already split in sentences
        concat = []
        for i,j in zip(english, german):
            concat.append(i + j)

        return concat

    def init_model(self, model = 'paraphrase-multilingual-mpnet-base-v2'):
        '''
        Initialize the desired model from SentenceTransformer. A list of the 
        available models can be find here https://www.sbert.net/docs/pretrained_models.html.
        Default is 'paraphrase-multilingual-mpnet-base-v2'.

        Parameters
        ----------
        model : string
            Name of the desired translation model, passed as a string.

        Returns
        -------
        model : model 
            Loads the passed model.

        '''
        model = SentenceTransformer(model)
        return model

    def embeddings_similarity(self, concatenation, model = 'paraphrase-multilingual-mpnet-base-v2'):
        '''
        Compute the embeddings and the cosine similiarity scores for each pair
        of document (i.e. the german and the english version of the same document).
        It ouputs for each pair a list of the sorted sentences-pair according
        to their cosine similarity score.

        Parameters
        ----------
        concatenation : list
            Concatenated list, is the output of the above concatenation function.

        model : string
            Name of the desired translation model, passed as a string. Default is
            'paraphrase-multilingual-mpnet-base-v2'. List of available models is linked in 
            documentation of init_models().

        Returns
        -------
        sorted_scores : list
            List of lists with the sentence pairs and the correspondents scores.

        '''
        # Initialize model
        model = self.init_model(model)

        # Compute embeddings and cosine similarity scores for each document pair
        scores = []
        embeds = []
        for sents in concatenation:
            embeddings = model.encode(sents)
            cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
            embeds.append(embeddings)
            scores.append(cosine_scores)

        # Concatenate each sentence pair with its cosine-similarity score
        pairs = []
        for scr in scores:
            pair = []
            for i in range(len(scr)-1):
                for j in range(i+1, len(scr)):
                    pair.append({'index': [i, j], 'score': scr[i][j]})
            pairs.append(pair)

        # Sort for each document the sentence pairs according to their score
        sorted_scores = []
        for line in pairs:
            sorted_scores.append(sorted(line, key = lambda x: x['score'], reverse = True))

        return sorted_scores

    def sim_df(self, concatenation,  sorted_scores):
        '''
        Create a pandas Datframe of the sentence pairs and their cosine
        similarity score.

        Parameters
        ----------
        concatenation : list
            Concatenated list, is the output of the above concatenation function.
            
        sorted_scores : list
            List of sentence pairs sorted according to their score. It is the 
            output of the above embeddings_similarity function.

        Returns
        -------
        df : dataframe
            Pandas dataframe of sentences and scores.

        '''
        sentence_german, sentence_english, score = [], [], []

        for doc, row in zip(concatenation, sorted_scores):
            for ind in row:
                i, j = ind['index']
                sentence_english.append(doc[i])
                sentence_german.append(doc[j])
                score.append(ind['score'])

        sentence_german = pd.Series(sentence_german)
        sentence_english = pd.Series(sentence_english)
        score = pd.Series(score)

        df = pd.DataFrame()
        df['deu'] = sentence_german
        df['eng'] = sentence_english
        df['score'] = [float(score) for score in score]

        return df

    def cosine_similarity(self, data, model = 'paraphrase-multilingual-mpnet-base-v2'):
        '''
        Function that computes the embeddings for the single sentences and returns
        the german-english pair with the highest cosine-similarity. For each
        embedding model is the used SBERT model. 

        Parameters
        ----------
        data : Dataframe
            Dataframe of the paired ad-hocs, generetaed with Import.findcounterpart().
            
        model : string, optional
            Name of the desired translation model, passed as a string. List of available models 
            is linked in documentation of init_models(). 
            The default is 'paraphrase-multilingual-mpnet-base-v2'.

        Returns
        -------
        df : Dataframe
            Pandas dataframe reporting the best translation for each sentence
            according to the cosine similarity.

        '''
        english = list(data['bodyText_x'])
        german = list(data['bodyText_y'])

        english_hashs = list(data['hash_x'])
        german_hashs = list(data['hash_y'])

        model = self.init_model(model)

        german_sen, english_sen, cosine_sim, eng_hashs, ger_hashs = [], [], [], [], []

        for eng, ger, eng_has, ger_has in zip(english, german, english_hashs, german_hashs):

            embeddings_eng = model.encode(eng)
            embeddings_ger = model.encode(ger)

            cosine_similarity = pd.DataFrame(util.pytorch_cos_sim(embeddings_eng, embeddings_ger))

            german_ind, english_ind, max_score =  [], [], []
            for sentence in cosine_similarity:
                sentence_x = cosine_similarity[sentence]
                sentence_x = [float(score) for score in sentence_x]
                max_sim = max(sentence_x)
                german_ind.append(sentence)
                english_ind.append(sentence_x.index(max_sim))
                max_score.append(max_sim)

            english_sentence = []
            for idx in english_ind:
                english_sentence.append(eng[idx])

            german_sen.extend(ger)
            english_sen.extend(english_sentence)
            cosine_sim.extend(max_score)
            eng_hashs.extend([eng_has] * len(english_sentence))
            ger_hashs.extend([ger_has] * len(english_sentence))
        
        df = pd.DataFrame()
        df['German_sentences'] = pd.Series(german_sen)
        df['English_sentences'] = pd.Series(english_sen)
        df['Cosine_score'] = pd.Series(cosine_sim)
        df['English_hash'] = pd.Series(eng_hashs)
        df['German_hash'] = pd.Series(ger_hashs)
        
        return df
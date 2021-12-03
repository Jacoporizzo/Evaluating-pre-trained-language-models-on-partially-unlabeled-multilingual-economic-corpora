# Import preprocessed data
import pickle 
import pandas as pd

english_goldstandards = pickle.load(open('data/english_goldstandards.pkl', 'rb'))
cosines_scores = pickle.load(open('data/cosine_scores.pkl', 'rb'))

# Clean dataset from same rows (bug-fix)
prova = english_goldstandards.drop_duplicates(['Sentences', 
                                               'English_sentences', 
                                               'Cosine_score', 
                                               'LabelsString'])

cosineshigh = (english_goldstandards['Cosine_score'] > 0.9)
cosineshigh.sum()

cosinesaccep = (english_goldstandards['Cosine_score'] > 0.8)
cosinesaccep.sum()

cosinesborder = (english_goldstandards['Cosine_score'] > 0.7)
cosinesborder.sum()

# Example borderline for cosine 0.7579
english_goldstandards['Sentences'][23]
english_goldstandards['English_sentences'][23]
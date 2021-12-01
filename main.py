# Import preprocessed data
import pickle 

english_goldstandards = pickle.load(open('data/english_goldstandards.pkl', 'rb'))
cosines_scores = pickle.load(open('data/cosine_scores.pkl', 'rb'))
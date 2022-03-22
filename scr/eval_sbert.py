'''
Evaluation of the output from the SBERT model for
the final df with the labelled english sentences.
'''
import pickle 

# Import data
cosines_scores = pickle.load(open('data/cosine_scores.pkl', 'rb'))
english_labelled = pickle.load(open('data/english_goldstandards.pkl', 'rb'))

# Get data above a given threshold
cosines_high = (english_labelled['Cosine_score'] > 0.9)
cosines_acceptable = (english_labelled['Cosine_score'] > 0.8)
cosines_borderline = (english_labelled['Cosine_score'] > 0.7)
cosines_bad = (english_labelled['Cosine_score'] < 0.7)

# Number of observation bove thresholds
print('# Goldstandards with a cosine similarity of' + '\n' +
      '>= 0,9: {}'.format(cosines_high.sum()) + '\n' +
      '>= 0,8: {}'.format(cosines_acceptable.sum()) + '\n' +
      '>= 0,7: {}'.format(cosines_borderline.sum()) + '\n' +
      '< 0,7: {}'.format(cosines_bad.sum()))

# Example for a borderline example
english_labelled['Cosine_score'][661]
english_labelled['German_goldstandards'][661]
english_labelled['English_sentences'][661]

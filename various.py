#%%
from sentence_transformers import SentenceTransformer, util
from utils.imports import Import

imp = Import()
data = imp.findcounterpart()
eng = data['bodyText_x'][103]
deu = data['bodyText_y'][103]

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Other models to try, reference https://www.sbert.net/docs/pretrained_models.html
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
#model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Concatenate lists
sentences = eng + deu

# Compute the embeddings
embeddings = model.encode(sentences)#, convert_to_tensor = True) 

# Compute the cosine-similarities for each sentence with another sentence
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

# Find pairs with highest similarity
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
        
# Sort scores in decreasing order
pairs = sorted(pairs, key = lambda x: x['score'], reverse = True)

#%% Working part

# Import data and concatenate the english and german version of the documents,
# i.e. both already split in sentences
from sentence_transformers import SentenceTransformer, util
from utils.imports import Import
import pandas as pd

imp = Import()
data = imp.findcounterpart()
test_df = data[0:10]
eng = list(test_df['bodyText_x'])
deu = list(test_df['bodyText_y'])
concat = []
for i, j in zip(eng, deu):
    concat.append(i + j)

# Initialize SBERT model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Compute embeddings and cosine similarities for each document pair 
scores = []
embeds = []
for sents in concat:
    embeddings = model.encode(sents)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    embeds.append(embeddings)
    scores.append(cosine_scores)

# Concatenate each pair with its cosine-similarity score
pairs = []
for scr in scores:
    pair = []
    for i in range(len(scr)-1):
        for j in range(i+1, len(scr)):
            pair.append({'index': [i, j], 'score': scr[i][j]})
    pairs.append(pair)

# Create sorted pair separately for each pair in pairs 
sorted_scores = []
for line in pairs:
    sorted_scores.append(sorted(line, key = lambda x: x['score'], reverse = True))

# Create a pandas DF containing all the needed info
sentence_german, sentence_english, score = [], [], []

for doc, row in zip(concat, sorted_scores):
    for ind in row:
        i, j = ind['index']
        sentence_german.append(doc[j])
        sentence_english.append(doc[i])
        score.append(ind['score'])

#sentence_german = pd.Series(sentence_german)
#sentence_english = pd.Series(sentence_english)
#score = pd.Series(score)

df = pd.DataFrame()
df['deu'] = pd.Series(sentence_german)
df['eng'] = pd.Series(sentence_english)
df['score'] = pd.Series([float(score) for score in score])

#%%
from utils.translations import Translation
import numpy as np
import pandas as pd
from utils.imports import Import
from utils.labels import Label

imp = Import()
data = imp.findcounterpart()
eng = data['bodyText_x'][0:10]
deu = data['bodyText_y'][0:10]

trans = Translation()
df = trans.cosine_similarity(eng, deu)

goldstd = imp.importgold()
control_df = goldstd[goldstd['Hashs'].isin(data['hash_y'][0:10])]

cross_val = df[df['German_sentences'].isin(control_df['Sentences'])]

label = Label()

select_df = label.index_container(goldstd, df)

# Problems with split 
empty_vector = [str('Empty') for x in np.arange(27)]
sen_idx = []
for idx in lst1:
    try:
        sen_idx.append(goldstd.iloc[idx])
    except:
        sen_idx.append(pd.Series(empty_vector))
        
prova1 = pd.DataFrame(sen_idx, columns = goldstd.columns).reset_index(drop = True)
df1 = df.copy()

df1 = df1[~pd.isna(df1['Sentences'])]
prova2 = prova1[~pd.isna(prova1['Sentences'])]

# Join the english translation and the cosine simil to each german sentence
final_df = prova2.join([df1['english'], df1['cosine_score']]).reset_index(drop = True)

# Try to join english hash
english_hash = []
for hashs in final_df['Hashs']:
    english_hash.append(data['hash_x'].where(hashs == data['hash_y']))

eng_test = [list(hashs.dropna()) for hashs in english_hash]

eng_test1 = []
for code in eng_test:
    if code != []:
        eng_test1.append(code)
    else:
        eng_test1.append(['Bash does not exist'])

hash_series = pd.Series([item for hashs in eng_test1 for item in hashs])

final_df['english_hash'] = hash_series
#%%
from utils.translations import Translation
import numpy as np
import pandas as pd
from utils.imports import Import

imp = Import()
data = imp.findcounterpart()
eng = data['bodyText_x'][0:10]
deu = data['bodyText_y'][0:10]

trans = Translation()
df = trans.cosine_similarity(eng, deu)
df1 = df.copy()

goldstd = imp.importgold()

gold_sentences = goldstd['Sentences']
single_sentences = df['German_sentences']
indices = []

df1.rename(columns = {'German_sentences': 'Sentences'}, inplace = True)
test_df = df1.merge(goldstd, how = 'left', on = 'Sentences')

test_df1 = df1.merge(goldstd, how = 'inner', on = 'Sentences')

eng_hashs = []
for has in test_df1['Hashs']:
    eng_hashs.append(data[data['hash_y'] == has]['hash_x'].values)

engs = pd.Series([item for has in eng_hashs for item in has])

test_df1['Hashs'] = engs
test_df1 = test_df1.drop(['Sentences', 'Cosine_score'], axis = 1)

singsent_index = np.where(single_sentences.isin(gold_sentences))[0]
gs_index = np.where(gold_sentences.isin(single_sentences))[0]

test = np.where(gold_sentences.str.contains('Mit der Heliad-Aktie profitieren private und institutionelle Investoren somit mittels eines täglich liquiden Dividendentitels von den Chancen eines diversifizierten Portfolios der interessantesten disruptiven Wachstumsunternehmen im deutschsprachigen Raum.'))[0]


testlst = []     

for boolean in single_sentences.isin(gold_sentences):
    if boolean:
        testlst.append(np.where(gold_sentences.isin(single_sentences))[0])
    else:
        testlst.append(['Empty'])

for sentence in single_sentences:
    indices.append(np.where(gold_sentences.str.contains(sentence, case = False, regex = False))[0])

gold_sentences = goldstd['Sentences']
single_sentences = df['German_sentences']
indices = []

for sentence in df['German_sentences']:
    indices.append(str(np.where(goldstd['Sentences'].str.contains(sentence, regex = False))[0]))

sum([len(i) != 2 for i in indices])

indices = []

for sentence in df['German_sentences']:
    indices.append(np.where(goldstd['Sentences'].str.contains(sentence, regex = False))[0])

#%%
from utils.translations import Translation
from utils.imports import Import

imp = Import()
data = imp.findcounterpart()
eng = data['bodyText_x'][0:10]
deu = data['bodyText_y'][0:10]

trans = Translation()
df = trans.cosine_similarity(eng, deu)
df1 = df.copy()

goldstd = imp.importgold()

gold_lst = list(goldstd['Sentences'])
sen_lst = list(df['German_sentences'])

indices = [s for s in gold_lst for sen in sen_lst if sen in s]
for sentence in sen_lst:
    indices.append(gold_lst.str.contains(sentence, regex = False))
    
indices = []
for sentence in sen_lst: # range(len(sen_lst)):
    if sentence in gold_lst:
        indices.append(sentence)
    else:
        indices.append('Not a gold standard')
        
goldstd_copy = goldstd.copy()        
goldstd = goldstd_copy.reset_index(drop = True)

gold_sentences = goldstd['Sentences']

indices = []
for sentence in df['German_sentences']:
    if (gold_sentences.str.contains(sentence, regex = False)).any():
        idx = np.where(gold_sentences.str.contains(sentence, regex = False))[0][0]
        indices.append(gold_sentences[idx])
    else:
        indices.append('Not a gold standard')

import pandas as pd            
df['German_sentences'] = pd.Series(indices)        
df.rename(columns = {'German_sentences': 'Sentences'}, inplace = True)

df_merge = df.merge(goldstd, how = 'inner', on = 'Sentences')

eng_hashs = []
for has in df_merge['Hashs']:
    eng_hashs.append(data[data['hash_y'] == has]['hash_x'].values)
    
#%%
goldtest = goldstd[goldstd['Sentences'].isin(df1['German_sentences'])].reset_index(drop = True)
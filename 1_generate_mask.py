#Generate Maskings
import csv
import spacy
import re
import tqdm
alpha = 0.001  
N=100000
#Measures the similiarity of sensorial representations
import math
import pickle
from scipy import spatial
import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt
import csv 
import random
from nltk.stem import WordNetLemmatizer
from collections import Counter
from LIWC import liwc
from nltk.corpus import cmudict
import math
CMUdict = cmudict.dict()      #syllable

lemmatizer = WordNetLemmatizer()
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from PassivePySrc import PassivePy

subtlex = {}
passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")

ANEW_valence = {}
ANEW_dominance = {}
ANEW_arousal = {}
with open('dict/anew.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            ANEW_valence[row[0].lower()]=float(row[2])/10
            ANEW_arousal[row[0].lower()]=float(row[4])/10
            ANEW_dominance[row[0].lower()]=float(row[6])/10
        line_count += 1
        

with open('dict/subtlex.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            #subtlex[row[0].lower()] = math.log10(8388/int(row[2]))     #Doc
            subtlex[row[0].lower()] = int(row[1])       #Freq
        line_count += 1

wrad = {}       
with open('dict/WRAD.Wt.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        
        wrad[row[0].lower()]=(float(row[1])+1)/2

attr_ = {}      #Attributes

subtlex = {}

nlp = spacy.load('en_core_web_lg')
def pair_list(list_):
    return[' '.join(list_[i:i+2]) for i in range(0, len(list_), 2)]

#load sense:
senses = []
with open('lexicon_weight.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        senses.append(row[0])

i = 0
sense_header = {}
for item in senses:
    sense_header[item] = i
    i = i+1

file = 'yelp_train_set_raw.csv'


print(file)
p = 0
data_sents = []
with open(file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in tqdm.tqdm(csv_reader):
        p=p+1
        if p<N:
            doc = nlp(row[0])
            sent_set = []
            for sent in doc.sents:
                new_string = re.sub(r'[^\w\s]','',str(sent).lower())
                sent_set.append(new_string)
            data_sents.append([sent_set,row[1]])

print(len(data_sents))

text = []
features = []
characteristics = []
row = ['PASSIVE VOICE', 'REFRENTIAL ACTIVITY', 'DOMINANCE', 'VALENCE','AROUSAL','REPETITION', 'AFFECT', 'POS_EMO','NEG_EMO','ANXIETY', 'ANGER', 'SADNESS','FUNCTION WORDS', 'PRONOUN', 'PERSONAL PRONOUN', '1PS PRONOUNS', '1PP PRONOUNS', '2P PRONOUNS', '3PS PRONOUNS', '3PP PRONOUNS', 'IMPERSONAL PRONOUNS', 'ARTICLES', 'PREPOSITIONS','AUXILIARY VERB', 'ADVERBS', 'CONJUNCTIONS', 'NEGATIONS', 'VERBS', 'ADJECTIVE', 'COMPARATIVE', 'INTERROGATIVE', 'SENTIMENT NEGATIVE', 'SENTIMENT POSITIVE', 'SENTIMENT NEUTRAL', 'SENTIMENT COMPOUND', 'READABILITY', 'SICHEL S', 'BRUNET W', 'HONORE R', 'TTR', 'REVIEWS']
characteristics.append(row)
features.append(senses)
text.append('text_set')
print('sentences',len(data_sents))
for item in tqdm.tqdm(data_sents):
    #print(item[0])
    feat_row = [0]*len(senses)
    candidate_set = []
    word_set = ' '.join(item[0]).replace('  ',' ').strip(' ').split(' ')

    text.append(' '.join(item[0]))
    candidate_set.extend(word_set)
    #candidate_set.extend(pair_list(word_set))
    #candidate_set.extend(pair_list(word_set[1:]))
    sense_words = list(set(candidate_set).intersection(set(senses)))
    if len(sense_words)>0:
        denom = 0
        for word in sense_words:
            denom = denom + word_set.count(word)

        for word in sense_words:
            TF = word_set.count(word)/denom
            feat_row[sense_header[word]]=TF

        features.append(feat_row)

        char_row = []
        voice_num = 0
        voice_denom = len(item[0])

        sent_score_neg = 0
        sent_score_pos = 0
        sent_score_neu = 0
        sent_score_compound = 0

        for sent in item[0]:
            if len(sent)>1:
                voice_ident = passivepy.match_text(sent, full_passive=True, truncated_passive=True)
                sent_score = sia.polarity_scores(sent)
                voice = sum(voice_ident['passive_sents_count'])
                voice_num += voice

                sent_score_neg = sent_score_neg + sent_score['neg']
                sent_score_pos = sent_score_pos + sent_score['pos']        
                sent_score_neu = sent_score_neu + sent_score['neu']        
                sent_score_compound = sent_score_compound + sent_score['compound']        

        passive = voice_num/voice_denom
        char_row.append(passive)

        wrad_num = 0
        wrad_denom = 0

        dominance_num = 0
        dominance_denom = 0

        valence_num = 0
        valence_denom = 0

        arousal_num = 0
        arousal_denom = 0
        syllable_count = 0
        repetition = len(word_set)-len(set(word_set))
        for word in word_set:
            if word in CMUdict:
                syllable_count += len(CMUdict[word][0])
            if word in wrad:
                wrad_num += wrad[word]
                wrad_denom +=1
            if word in ANEW_dominance:
                dominance_num += ANEW_dominance[word]
                dominance_denom += 1

                valence_num += ANEW_valence[word]
                valence_denom += 1

                arousal_num += ANEW_arousal[word]
                arousal_denom += 1
        if wrad_denom>0: 
            char_row.append(wrad_num/wrad_denom)
        else:
            char_row.append(0)

        if dominance_denom>0: 
            char_row.append(dominance_num/dominance_denom)    
        else:
            char_row.append(0)


        if valence_denom>0: 
            char_row.append(valence_num/valence_denom)        
        else:
            char_row.append(0)

        if arousal_denom>0: 
            char_row.append(arousal_num/arousal_denom)        
        else:
            char_row.append(0)

        char_row.append(repetition/len(word_set))        

        liwcArray=liwc().getLIWCCount(' '.join(item[0]))


        #Affect
        char_row.append(liwcArray['affect']/liwcArray['WC'])
        char_row.append(liwcArray['posemo']/liwcArray['WC'])
        char_row.append(liwcArray['negemo']/liwcArray['WC'])    
        char_row.append(liwcArray['anx']/liwcArray['WC'])

        char_row.append(liwcArray['anger']/liwcArray['WC'])
        char_row.append(liwcArray['sad']/liwcArray['WC'])    


        #Function Words

        char_row.append(liwcArray['funct']/liwcArray['WC'])
        char_row.append(liwcArray['pronoun']/liwcArray['WC'])
        char_row.append(liwcArray['ppron']/liwcArray['WC'])
        char_row.append(liwcArray['i']/liwcArray['WC'])    
        char_row.append(liwcArray['we']/liwcArray['WC'])
        char_row.append(liwcArray['you']/liwcArray['WC'])
        char_row.append(liwcArray['shehe']/liwcArray['WC'])
        char_row.append(liwcArray['they']/liwcArray['WC'])    

        char_row.append(liwcArray['ipron']/liwcArray['WC'])
        char_row.append(liwcArray['article']/liwcArray['WC'])
        char_row.append(liwcArray['prep']/liwcArray['WC'])    
        char_row.append(liwcArray['auxverb']/liwcArray['WC'])

        char_row.append(liwcArray['adverb']/liwcArray['WC'])
        char_row.append(liwcArray['conj']/liwcArray['WC'])
        char_row.append(liwcArray['negate']/liwcArray['WC'])    
        char_row.append(liwcArray['verb']/liwcArray['WC'])

        char_row.append(liwcArray['adj']/liwcArray['WC'])
        char_row.append(liwcArray['compare']/liwcArray['WC'])
        char_row.append(liwcArray['interrog']/liwcArray['WC'])    



        char_row.append(sent_score_neg/liwcArray['WC'])    
        char_row.append(sent_score_pos/liwcArray['WC'])        
        char_row.append(sent_score_neu/liwcArray['WC'])        
        char_row.append(sent_score_compound/liwcArray['WC'])        

        unique_words = len(set(word_set))
        hapax= Counter(word_set)
        syllables = syllable_count

        FK = 206.835 - 1.015*(liwcArray['WC']/len(item[0]))-84.6*(syllables/liwcArray['WC'])


        num=liwcArray['WC']
        words=liwcArray['WC']    
        cnt=0
        cnt2 = 0
        for w in hapax:
            if hapax[w]==2:
                cnt=cnt+1
            if hapax[w]==1:
                cnt2 += 1
        S = float(cnt)/float(len(hapax))
        a=0.172
        W = words** (len(hapax) **a)

        num=words
        cnt=0
        
        denom=1-(float(cnt2)/float(len(hapax)+alpha))
        
        R = 100*math.log10(num/denom)
        TTR = len(hapax)/words

        char_row.append(FK)
        char_row.append(S)        
        char_row.append(W)    
        char_row.append(R)    
        char_row.append(TTR)    
        char_row.append(int(item[1])/5)    

        characteristics.append(char_row)


with open('feature_matrix.csv', mode='w') as _file:
    writer = csv.writer(_file, delimiter=',')
    writer.writerows(features)    


with open('characteristic_matrix.csv', mode='w') as _file:
    writer = csv.writer(_file, delimiter=',')
    writer.writerows(characteristics)





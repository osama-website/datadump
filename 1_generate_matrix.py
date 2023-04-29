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

passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")

WIN = 40
attr_ = {}      #Attributes

subtlex = {}

#load sense:
sense_headers = []
senses = {}
sense_strength = {}
sense_rarity = {}
rarity_counter = 0
with open('dict/lexicon_weight.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        sense_headers.append(row[1])
        senses[row[0]]=row[1]
        sense_strength[row[0]] = float(row[2])/5
        if row[0] in subtlex:
            rarity_counter += subtlex[row[0]]

sense_headers=list(set(sense_headers))
sense_headers.sort()
for word in senses:
    try:
        sense_rarity[word] = subtlex[word]/rarity_counter
    except:    
        sense_rarity[word] = 0


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


def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


file = 'yelp_train_set_mask.csv'
data = []
i=0
with open(file,encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in tqdm.tqdm(csv_reader):
        item = {}
        i=i+1
        sent = row[2].replace(' <MASK>','')
        if len(sent.split(' '))>3:
            voice_ident = passivepy.match_text(sent, full_passive=True, truncated_passive=True)
            voice = sum(voice_ident['passive_sents_count'])
            item['text'] = row[0]
            item['attr_text'] = sent
            item['mask_text'] = row[2]
            #WRAD
            wrad_num = 0
            wrad_denom = 0

            dominance_num = 0
            dominance_denom = 0

            valence_num = 0
            valence_denom = 0

            arousal_num = 0
            arousal_denom = 0
            syllable_count = 0
            repetition = len(sent.split(' '))-len(set(sent.split(' ')))
            for word in sent.split(' '):
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

            item['attr'] = {}
            item['feat'] = {}
            item['feat']['word'] = row[3]
            item['feat']['sense'] = senses[row[3]]
            item['feat']['strength'] = sense_strength[row[3]]
            item['feat']['rarity'] = sense_rarity[row[3]]
            item['attr']['wrad']=[wrad_num,wrad_denom,'REFRENTIAL ACTIVITY']

            liwcArray=liwc().getLIWCCount(sent)
            #Affect
            item['attr']['affect'] = [liwcArray['affect'],liwcArray['WC'],'AFFECT']
            item['attr']['pos_emo'] = [liwcArray['posemo'],liwcArray['WC'],'AFFECT']
            item['attr']['neg_emo'] = [liwcArray['negemo'],liwcArray['WC'],'AFFECT']
            item['attr']['anxiety'] = [liwcArray['anx'],liwcArray['WC'],'AFFECT']
            item['attr']['anger'] = [liwcArray['anger'],liwcArray['WC'],'AFFECT']
            item['attr']['sadness'] = [liwcArray['sad'],liwcArray['WC'],'AFFECT']

            #Function Words
     
            item['attr']['function_words'] = [liwcArray['funct'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['pronoun'] = [liwcArray['pronoun'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['personal_pronouns'] = [liwcArray['ppron'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['1ps_pronouns'] = [liwcArray['i'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['1pp_pronouns'] = [liwcArray['we'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['2p_pronouns'] = [liwcArray['you'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['3ps_pronouns'] = [liwcArray['shehe'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['3pp_pronouns'] = [liwcArray['they'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['impersonal_pronouns'] = [liwcArray['ipron'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['articles'] = [liwcArray['article'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['prepositions'] = [liwcArray['prep'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['auxiliary_verb'] = [liwcArray['auxverb'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['adverbs'] = [liwcArray['adverb'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['conjunctions'] = [liwcArray['conj'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['negations'] = [liwcArray['negate'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['verbs'] = [liwcArray['verb'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['adjective'] = [liwcArray['adj'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['comparative'] = [liwcArray['compare'],liwcArray['WC'],'FUNCTION WORDS']
            item['attr']['interrogative'] = [liwcArray['interrog'],liwcArray['WC'],'FUNCTION WORDS']
            
            sent_score = sia.polarity_scores("Wow, NLTK is really powerful!")
            #SENTIMENT
            item['attr']['neg_sent'] = [sent_score['neg'],liwcArray['WC'],'SENTIMENT']
            item['attr']['pos_sent'] = [sent_score['pos'],liwcArray['WC'],'SENTIMENT']
            item['attr']['neutral_sent'] = [sent_score['neu'],liwcArray['WC'],'SENTIMENT']
            item['attr']['compound_sent'] = [sent_score['compound'],liwcArray['WC'],'SENTIMENT']

            #VALENCE
            item['attr']['arousal'] = [arousal_num,arousal_denom,'VALENCE']
            item['attr']['valence'] = [valence_num,valence_denom,'VALENCE']
            item['attr']['dominance'] = [dominance_num,dominance_denom,'VALENCE']

            #SYNTAX
            item['attr']['repetition'] = [repetition,liwcArray['WC'],'SYNTAX']
            item['attr']['passive_voice'] = [voice,1,'SYNTAX']

            #Readability features:
            item['richness'] = {}
            item['richness']['words'] = liwcArray['WC']
            item['richness']['sentences'] = 1
            item['richness']['syllables'] = syllable_count
            item['richness']['unique_words'] = len(set(sent.split(' ')))
            item['richness']['hapax']= Counter(sent.split(' '))
            print(row)
            if i>10:
                print(XXX)
            data.append(item)

dbfile = open('yelp.attr', 'ab')
      
# source, destination
pickle.dump(data, dbfile)                     
dbfile.close()        
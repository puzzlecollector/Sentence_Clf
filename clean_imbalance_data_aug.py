import numpy as np 
import pandas as pd 
import pickle 
from tqdm.auto import tqdm
import re 

df = pd.read_csv("train.csv") 

types = df["유형"].values 
polarity = df["극성"].values 
tense = df["시제"].values 
certainty = df["확실성"].values 

sentences = df["문장"].values 
# backtranslated_sentences_dict_eng.pkl

with open("backtranslated_sentences_dict_eng.pkl", "rb") as pkl_dict:
    eng_aug_dict = pickle.load(pkl_dict) 

only_imbalanced_dict = {} 

for i in tqdm(range(len(sentences)), position=0, leave=True): 
    if sentences[i] in eng_aug_dict.keys(): 
        if (types[i] in ["추론형", "대화형", "예측형"]) or (polarity[i] in ["부정", "미정"]) or (tense[i] in ["미래"]) or (certainty[i] in ["불확실"]): 
            only_imbalanced_dict[sentences[i]] = eng_aug_dict[sentences[i]] 

# HTML 문자열 지우기 
# 패턴은 관찰해본 결과 3가지 밖에 없음 
clean_aug_dict = {} 

for k, v in tqdm(only_imbalanced_dict.items(), position=0, leave=True): 
    #mod_v = re.sub("&#39;", "\'", v) 
    #mod_v = re.sub("&quot;", "\"", mod_v) 
    mod_v = v.replace("&#39;", "\'") 
    mod_v = mod_v.replace("&quot;", "\"") 
    mod_v = mod_v.replace("&amp;", "&") 
    clean_aug_dict[k] = mod_v

with open("clean_imbalance_aug_dict.pkl", "wb") as handle: 
    pickle.dump(clean_aug_dict, handle)

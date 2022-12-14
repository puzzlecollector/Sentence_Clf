import numpy as np 
import random 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 

from tqdm.auto import tqdm 
from transformers import * 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 

import time 
from datetime import datetime 

from sklearn.metrics import f1_score 
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

# naive train/test split 
df = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv") 

train, val, _, _ = train_test_split(df, df["label"], test_size=0.2)

# simple data preprocessing 
train_sentences = train["문장"].values 
train_Y1 = train["유형"].values 
train_Y2 = train["극성"].values 
train_Y3 = train["시제"].values 
train_Y4 = train["확실성"].values 

val_sentences = val["문장"].values 
val_Y1 = val["유형"].values 
val_Y2 = val["극성"].values 
val_Y3 = val["시제"].values 
val_Y4 = val["확실성"].values 

Y1_dict = {"대화형": 0, "사실형": 1, "예측형": 2, "추론형": 3} 
Y2_dict = {"긍정": 0, "미정": 1, "부정": 2}
Y3_dict = {"과거": 0, "미래": 1, "현재": 2} 
Y4_dict = {"불확실": 0, "확실": 1} 

Y1_dict_rev = {0: "대화형", 1: "사실형", 2: "예측형", 3: "추론형"} 
Y2_dict_rev = {0: "긍정", 1: "미정", 2: "부정"} 
Y3_dict_rev = {0: "과거", 1: "미래", 2: "현재"} 
Y4_dict_rev = {0: "불확실", 1: "확실"} 

train_Y1_cat = [] 
for i in range(len(train_Y1)): 
    train_Y1_cat.append(Y1_dict[train_Y1[i]]) 
    
train_Y2_cat = [] 
for i in range(len(train_Y2)):
    train_Y2_cat.append(Y2_dict[train_Y2[i]]) 

train_Y3_cat = [] 
for i in range(len(train_Y3)): 
    train_Y3_cat.append(Y3_dict[train_Y3[i]])

train_Y4_cat = [] 
for i in range(len(train_Y4)): 
    train_Y4_cat.append(Y4_dict[train_Y4[i]]) 

val_Y1_cat = [] 
for i in range(len(val_Y1)): 
    val_Y1_cat.append(Y1_dict[val_Y1[i]]) 

val_Y2_cat = [] 
for i in range(len(val_Y2)): 
    val_Y2_cat.append(Y2_dict[val_Y2[i]]) 

val_Y3_cat = [] 
for i in range(len(val_Y3)): 
    val_Y3_cat.append(Y3_dict[val_Y3[i]]) 

val_Y4_cat = [] 
for i in range(len(val_Y4)): 
    val_Y4_cat.append(Y4_dict[val_Y4[i]]) 
    
    
# define tokenizer 
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


# define model 
class MeanPooling(nn.Module): 
    def __init__(self): 
        super(MeanPooling, self).__init__() 
    def forward(self, last_hidden_state, attention_masks): 
        input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float() 
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) 
        sum_mask = input_mask_expanded.sum(1) 
        sum_mask = torch.clamp(sum_mask, min=1e-9) 
        mean_embeddings = sum_embeddings / sum_mask 
        return mean_embeddings 

class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples
    def forward(self, out): 
        return torch.mean(torch.stack([self.classifier(self.dropout(p=self.max_dropout_rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)
    
    
class NeuralCLF(nn.Module): 
    def __init__(self, plm="snunlp/KR-SBERT-V40K-klueNLI-augSTS", type_classes=4, polarity_classes=3, tense_classes=3, certainty_classes=2): 
        super(NeuralCLF, self).__init__() 
        self.type_classes = type_classes 
        self.polarity_classes = polarity_classes 
        self.tense_classes = tense_classes 
        self.certainty_classes = certainty_classes 
        self.config = AutoConfig.from_pretrained(plm) 
        self.model = AutoModel.from_pretrained(plm, config=self.config) 
        self.tokenizer = AutoTokenizer.from_pretrained(plm) 
        self.mean_pooler = MeanPooling() 
        
        self.type_fc = nn.Linear(self.config.hidden_size, self.type_classes) 
        self._init_weights(self.type_fc)
        self.type_multi_dropout = MultiSampleDropout(0.2, 8, self.type_fc) 
        
        self.polarity_fc = nn.Linear(self.config.hidden_size, self.polarity_classes) 
        self._init_weights(self.polarity_fc) 
        self.polarity_multi_dropout = MultiSampleDropout(0.2, 8, self.polarity_fc)  
        
        self.tense_fc = nn.Linear(self.config.hidden_size, self.tense_classes) 
        self._init_weights(self.tense_fc) 
        self.tense_multi_dropout = MultiSampleDropout(0.2, 8, self.tense_fc) 
        
        self.certainty_fc = nn.Linear(self.config.hidden_size, self.certainty_classes) 
        self._init_weights(self.certainty_fc) 
        self.certainty_multi_dropout = MultiSampleDropout(0.2, 8, self.certainty_fc) 
        
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) 
            if module.bias is not None: 
                module.bias.data.zero_()
                
    def forward(self, input_ids, attn_masks):
        x = self.model(input_ids, attn_masks)[0] 
        x = self.mean_pooler(x, attn_masks) 
        type_output = self.type_multi_dropout(x) 
        polarity_output = self.polarity_multi_dropout(x) 
        tense_output = self.tense_multi_dropout(x) 
        certainty_output = self.certainty_multi_dropout(x) 
        return type_output, polarity_output, tense_output, certainty_output 
    
class WeightedFocalLoss(nn.Module): 
    def __init__(self, alpha, gamma=2): 
        super(WeightedFocalLoss, self).__init__() 
        self.alpha = alpha 
        self.device = torch.device("cuda") 
        self.alpha = self.alpha.to(self.device)  
        self.gamma = gamma 
    def forward(self, inputs, targets): 
        CE_loss = nn.CrossEntropyLoss()(inputs, targets) 
        targets = targets.type(torch.long) 
        at = self.alpha.gather(0, targets.data.view(-1))  
        pt = torch.exp(-CE_loss) 
        F_loss = at * (1-pt)**self.gamma * CE_loss 
        return F_loss.mean() 
    
def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return np.sum(pred_flat == labels_flat) / len(labels_flat)  

test_sentences = test["문장"].values 

test_input_ids, test_attn_masks = [], [] 
for i in tqdm(range(len(test_sentences))):
    encoded_inputs = tokenizer(test_sentences[i], max_length=256, truncation=True, padding='max_length') 
    test_input_ids.append(encoded_inputs["input_ids"]) 
    test_attn_masks.append(encoded_inputs["attention_mask"]) 
    
test_input_ids = torch.tensor(test_input_ids, dtype=int) 
test_attn_masks = torch.tensor(test_attn_masks, dtype=int) 

batch_size = 32 
test_data = TensorDataset(test_input_ids, test_attn_masks) 
test_sampler = SequentialSampler(test_data) 
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)  

model = NeuralCLF()
checkpoint = torch.load("KRSBERT_Epoch2_val_loss:0.012967109306303497.pt") 
model.load_state_dict(checkpoint) 
model.eval() 
model.cuda() 


type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], [] 

for step, batch in tqdm(enumerate(test_dataloader), position=0, leave=True, total=len(test_dataloader)): 
    batch = tuple(t.to(device) for t in batch) 
    b_input_ids, b_input_mask = batch  
    with torch.no_grad(): 
        type_logit, polarity_logit, tense_logit, certainty_logit = model(b_input_ids, b_input_mask) 
    
    type_preds += type_logit.argmax(1).detach().cpu().numpy().tolist() 
    polarity_preds += polarity_logit.argmax(1).detach().cpu().numpy().tolist() 
    tense_preds += tense_logit.argmax(1).detach().cpu().numpy().tolist() 
    certainty_preds += certainty_logit.argmax(1).detach().cpu().numpy().tolist()

str_type_preds, str_polarity_preds, str_tense_preds, str_certainty_preds = [], [], [], [] 

for i in range(len(type_preds)): 
    str_type_preds.append(Y1_dict_rev[type_preds[i]]) 
    str_polarity_preds.append(Y2_dict_rev[polarity_preds[i]]) 
    str_tense_preds.append(Y3_dict_rev[tense_preds[i]]) 
    str_certainty_preds.append(Y4_dict_rev[certainty_preds[i]]) 
    
answers = [] 

for str_type_pred, str_polarity_pred, str_tense_pred, str_certainty_pred in zip(str_type_preds, str_polarity_preds, str_tense_preds, str_certainty_preds): 
    answers.append(str_type_pred + "-" + str_polarity_pred + "-" + str_tense_pred + "-" + str_certainty_pred) 
    
    
submission = pd.read_csv("sample_submission.csv") 
submission["label"] = answers 

submission.to_csv("KRSBERT_refined_baseline.csv",index=False) 

import numpy as np
import pandas as pd
import random
import os
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import time

from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import re

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class MultiSampleDropout(nn.Module):
    def __init__(self, max_dropout_rate, num_samples, classifier):
        super().__init__()
        self.max_dropout_rate = max_dropout_rate
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Sequential(
            classifier,
            nn.Dropout(p=i),
            ) for i in np.linspace(0, self.max_dropout_rate, self.num_samples)])
        
    def forward(self, out):
        return torch.mean(torch.stack([layer(out) for layer in self.dropouts], dim=0), dim=0)


class NeuralCLF(nn.Module):
    def __init__(self, plm="monologg/kobigbird-bert-base", msd=0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(plm)

        self.type_lm = AutoModel.from_pretrained(plm, config=self.config)
        self.polarity_lm = AutoModel.from_pretrained(plm, config=self.config)
        self.tense_lm = AutoModel.from_pretrained(plm, config=self.config)
        self.certainty_lm = AutoModel.from_pretrained(plm, config=self.config)

        self.type_fc = nn.Linear(self.config.hidden_size, 4, bias=False)
        self.type_multi_dropout = MultiSampleDropout(msd, 8, self.type_fc)

        self.polarity_fc = nn.Linear(self.config.hidden_size, 3, bias=False)
        self.polarity_multi_dropout = MultiSampleDropout(msd, 8, self.polarity_fc)

        self.tense_fc = nn.Linear(self.config.hidden_size, 3, bias=False)
        self.tense_multi_dropout = MultiSampleDropout(msd, 8, self.tense_fc)

        self.certainty_fc = nn.Linear(self.config.hidden_size, 2, bias=False)
        self.certainty_multi_dropout = MultiSampleDropout(msd, 8, self.certainty_fc)

        self.cat_linear = nn.Sequential(
            nn.Linear(self.config.hidden_size * 4, self.config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
        )

    def forward(self, input_ids, attn_masks):
        type_x = self.type_lm(input_ids, attn_masks)[0]
        type_x = self.mean_pooling(type_x, attn_masks)

        polarity_x = self.polarity_lm(input_ids, attn_masks)[0]
        polarity_x = self.mean_pooling(polarity_x, attn_masks)

        tense_x = self.tense_lm(input_ids, attn_masks)[0]
        tense_x = self.mean_pooling(tense_x, attn_masks)

        certainty_x = self.certainty_lm(input_ids, attn_masks)[0]
        certainty_x = self.mean_pooling(certainty_x, attn_masks)

        out_x = torch.cat((type_x, polarity_x, tense_x, certainty_x), dim=-1)
        out_x = self.cat_linear(out_x)
        
        type_output = self.type_multi_dropout(out_x) + self.type_multi_dropout(type_x)
        polarity_output = self.polarity_multi_dropout(out_x) + self.polarity_multi_dropout(polarity_x)
        tense_output = self.tense_multi_dropout(out_x) + self.tense_multi_dropout(tense_x)
        certainty_output = self.certainty_multi_dropout(out_x) + self.certainty_multi_dropout(certainty_x)
        return type_output, polarity_output, tense_output, certainty_output

    @staticmethod
    def mean_pooling(last_hidden_state, attention_masks):
        input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss()(inputs, targets)
        targets = targets.type(torch.long)
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss
        return F_loss.mean()

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def replace_html(my_str):
    parseText = re.sub("&quot;", "\"", my_str)
    return parseText

def create_folds(data, num_splits):
    data["kfold"] = -1
    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=8888)
    labels = ["문장", "유형", "극성", "시제", "확실성"]
    data_labels = data[labels].values
    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f
    return data

if __name__ == "__main__":
    #################################################################################################
    num_splits = 10
    learning_rate = 3e-5
    warmup_steps = 0
    loss_rate = [1, 1, 1, 1]
    msd = 0.2
    
    criterion = WeightedFocalLoss()  # nn.CrossEntropyLoss()
    
    '''
    plm = "monologg/kobigbird-bert-base"
    epochs = 5
    batch_size = 24
    gas = 8
    '''
    ''''''
    plm = "klue/roberta-large"
    epochs = 5
    batch_size = 12
    gas = 16
    ''''''
    
    seed_everything(7789)
    #################################################################################################
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    Y1_dict = {"대화형": 0, "사실형": 1, "예측형": 2, "추론형": 3}
    Y2_dict = {"긍정": 0, "미정": 1, "부정": 2}
    Y3_dict = {"과거": 0, "미래": 1, "현재": 2}
    Y4_dict = {"불확실": 0, "확실": 1}

    tokenizer = AutoTokenizer.from_pretrained(plm)

    df = pd.read_csv("train.csv")

    df = create_folds(df, num_splits)    

    for f in range(num_splits):
        print(f"===== validating on fold {f+1} =====")
        train, val = df[df["kfold"] != f], df[df["kfold"] == f]

        train_sentences = train["문장"].values
        val_sentences = val["문장"].values

        train_input_ids, train_attn_masks = [], []
        for i in tqdm(range(len(train_sentences))):
            encoded_inputs = tokenizer(train_sentences[i], max_length=400, truncation=True, padding='max_length')
            train_input_ids.append(encoded_inputs["input_ids"])
            train_attn_masks.append(encoded_inputs["attention_mask"])

        val_input_ids, val_attn_masks = [], []
        for i in tqdm(range(len(val_sentences))):
            encoded_inputs = tokenizer(val_sentences[i], max_length=400, truncation=True, padding='max_length')
            val_input_ids.append(encoded_inputs["input_ids"])
            val_attn_masks.append(encoded_inputs["attention_mask"])

        train_input_ids = torch.tensor(train_input_ids, dtype=int)
        train_attn_masks = torch.tensor(train_attn_masks, dtype=int)

        val_input_ids = torch.tensor(val_input_ids, dtype=int)
        val_attn_masks = torch.tensor(val_attn_masks, dtype=int)

        train_Y1_labels = torch.tensor([Y1_dict[i] for i in train["유형"].values], dtype=int)
        val_Y1_labels = torch.tensor([Y1_dict[i] for i in val["유형"].values], dtype=int)

        train_Y2_labels = torch.tensor([Y2_dict[i] for i in train["극성"].values], dtype=int)
        val_Y2_labels = torch.tensor([Y2_dict[i] for i in val["극성"].values], dtype=int)

        train_Y3_labels = torch.tensor([Y3_dict[i] for i in train["시제"].values], dtype=int)
        val_Y3_labels = torch.tensor([Y3_dict[i] for i in val["시제"].values], dtype=int)

        train_Y4_labels = torch.tensor([Y4_dict[i] for i in train["확실성"].values], dtype=int)
        val_Y4_labels = torch.tensor([Y4_dict[i] for i in val["확실성"].values], dtype=int)

        train_data = TensorDataset(train_input_ids, train_attn_masks, train_Y1_labels, train_Y2_labels, train_Y3_labels, train_Y4_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_input_ids, val_attn_masks, val_Y1_labels, val_Y2_labels, val_Y3_labels, val_Y4_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        best_f1 = 0

        model = NeuralCLF(plm=plm, msd=msd)
        model.cuda()
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = warmup_steps,
                                                    num_training_steps = total_steps)
        model.zero_grad()
        for epoch_i in tqdm(range(0,epochs), desc="Epochs", position=0, leave=True, total=epochs):
            train_loss = 0
            model.train()
            with tqdm(train_dataloader, unit = "batch") as tepoch:
                for step, batch in enumerate(tepoch):
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_type_labels, b_polarity_labels, b_tense_labels, b_certainty_labels = batch
                    type_logit, polarity_logit, tense_logit, certainty_logit = model(b_input_ids, b_input_mask)
                    loss = loss_rate[0] * criterion(type_logit, b_type_labels) + loss_rate[1] * criterion(polarity_logit, b_polarity_labels) + loss_rate[2] * criterion(tense_logit, b_tense_labels) + loss_rate[3] * criterion(certainty_logit, b_certainty_labels)
                    train_loss += loss.item()
                    loss.backward()
                    if step % gas == gas - 1 or step == len(train_dataloader)-1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        tepoch.set_postfix(loss=train_loss / (step+1))
                        time.sleep(0.1)
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"average train loss: {avg_train_loss}")
            val_loss = 0
            model.eval()
            type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
            type_labels, polarity_labels, tense_labels, certainty_labels = [], [], [], []
            for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_type_labels, b_polarity_labels, b_tense_labels, b_certainty_labels = batch
                with torch.no_grad():
                    type_logit, polarity_logit, tense_logit, certainty_logit = model(b_input_ids, b_input_mask)
                    loss = loss_rate[0] * criterion(type_logit, b_type_labels) + loss_rate[1] * criterion(polarity_logit, b_polarity_labels) + loss_rate[2] * criterion(tense_logit, b_tense_labels) + loss_rate[3] * criterion(certainty_logit, b_certainty_labels)
                    val_loss += loss.item()

                    type_preds += type_logit.argmax(1).detach().cpu().numpy().tolist()
                    type_labels += b_type_labels.detach().cpu().numpy().tolist()

                    polarity_preds += polarity_logit.argmax(1).detach().cpu().numpy().tolist()
                    polarity_labels += b_polarity_labels.detach().cpu().numpy().tolist()

                    tense_preds += tense_logit.argmax(1).detach().cpu().numpy().tolist()
                    tense_labels += b_tense_labels.detach().cpu().numpy().tolist()

                    certainty_preds += certainty_logit.argmax(1).detach().cpu().numpy().tolist()
                    certainty_labels += b_certainty_labels.detach().cpu().numpy().tolist()

            type_f1 = f1_score(type_labels, type_preds, average="weighted")
            polarity_f1 = f1_score(polarity_labels, polarity_preds, average="weighted")
            tense_f1 = f1_score(tense_labels, tense_preds, average="weighted")
            certainty_f1 = f1_score(certainty_labels, certainty_preds, average="weighted")

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"average val loss: {avg_val_loss}")
            print(f"유형 F1: {type_f1}")
            print(f"극성 F1: {polarity_f1}")
            print(f"시제 F1: {tense_f1}")
            print(f"확실성 F1: {certainty_f1}")
            f1_mult = type_f1 * polarity_f1 * tense_f1 * certainty_f1
            print(f"f1 mult : {f1_mult}")
            if f1_mult > best_f1:
                best_f1 = f1_mult
                torch.save(model.state_dict(), f"{plm.split('/')[-1]}_Fold{f+1}_f1:{f1_mult}.pt")

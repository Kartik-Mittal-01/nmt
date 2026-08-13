'''data preprocessing pipeline'''
import re
import pandas as pd
import torch
import unicodedata
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

'''clean the text by removing extra spaces and normalizing text'''
def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = re.sub(r' +', ' ', text).strip()
    text = unicodedata.normalize('NFC', text)
    return text


PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

class NMTDataset(Dataset):
    '''csv_path is the path for the dataset'''
    '''tgt_tokenizer is BPE tokenizer used to convert target sentences into token IDs'''
    '''src_tokenizer_mode is either glove for bi-LSTM encoder OR bert for BERT encoder architecture'''
    '''glove_word2idx is Dictionary mapping each source word to its GloVe token ID'''
    '''bert_tokenizer is used in case encoder stack is BERT architecture ... converts source sentences into BERT token IDs'''
    def __init__(self, csv_path, tgt_tokenizer, src_tokenizer_mode="glove",
                 glove_word2idx=None, bert_tokenizer=None,
                 max_src_len=128, max_tgt_len=128, SVO_reversal=False, is_train=False):
        '''read the dataset'''
        self.df = pd.read_csv(csv_path)
        '''clean the dataset i.e, normalize and remove extra spaces'''
        self.df["source"] = self.df["source"].astype(str).str.replace('translate to Hindi: ', '').str.replace('translate to Marathi: ', '')
        self.df["source"] = self.df["source"].apply(normalize_text)
        self.df["translated"] = self.df["translated"].astype(str).apply(normalize_text)
        '''if we are working with train dataset make sure only to take entries where sentence length are <= 100 to maintain training efficiency'''
        if is_train:
            src_len = self.df["source"].apply(lambda x: len(x.split()))
            tgt_len = self.df["translated"].apply(lambda x: len(x.split()))
            self.df = self.df[(src_len <= 100) & (tgt_len <= 100)].reset_index(drop=True)
        '''glove or bert'''
        self.src_mode = src_tokenizer_mode
        '''tgt_tokenizer is BPE tokenizer trained on hindi and marathi dataset'''
        self.tgt_tokenizer = tgt_tokenizer
        '''in case glove src_tokenizer_mode is used'''
        self.word2idx = glove_word2idx
        '''Used when the source tokenizer mode is BERT'''
        self.bert_tokenizer = bert_tokenizer
        '''maximum sentence length for input sentences = 128'''
        self.max_src_len = max_src_len
        '''maximum sentence length for output sentences = 128'''
        self.max_tgt_len = max_tgt_len
        '''preoptimization technique for reversing input sentences'''
        self.SVO_reversal = SVO_reversal

    def __len__(self):
        '''how many samples are in your dataset?'''
        return len(self.df)
        
    '''Encodes the source sentence into GloVe token IDs for the 2-layer Bi-LSTM encoder'''
    def _encode_src_glove(self, text):
        '''Convert sentence to lowercase, split into words, and truncate to max length'''
        tokens = str(text).lower().split()[:self.max_src_len]
        if self.SVO_reversal:
            '''Optionally reverse the source token sequence'''
            tokens = tokens[::-1]
        '''Convert words to GloVe token IDs (use <unk> for OOV i.e, Out Of Vocabulary words); these IDs are later mapped to 100-D GloVe embeddings'''
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        '''Ensure the source sequence contains at least one token'''
        if not ids:
            ids = [UNK_IDX]
        '''Return GloVe input IDs and attention mask'''
        return torch.tensor(ids, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)

    '''Encodes the source sentence into BERT token IDs for the BERT encoder Architecture'''
    '''add_special_tokens=True adds [CLS] at the beginning and [SEP] at the end'''
    def _encode_src_bert(self, text):
        enc = self.bert_tokenizer(
            str(text), truncation=True, max_length=self.max_src_len,
            return_tensors="pt", add_special_tokens=True,
        )
        '''Return BERT input IDs and attention mask'''
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    '''Encodes the target sentence into BPE token IDs for the 2 Layer LSTM decoder Architecture'''
    def _encode_tgt(self, text):
        '''Tokenize the target sentence using the trained BPE tokenizer'''
        encoded = self.tgt_tokenizer(str(text), add_special_tokens=False)
        '''token ids for the target sentence'''
        ids = encoded["input_ids"]
        '''Convert token IDs to a Python list if returned as a tensor'''
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        '''Keep only the first max_tgt_len - 2 tokens so there is room to add <SOS> and <EOS>.'''
        ids = ids[:self.max_tgt_len - 2]
        '''Add <SOS> at the beginning and <EOS> at the end of the target sequence'''
        ids = [SOS_IDX] + ids + [EOS_IDX]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        '''get the row at index 'idx' in the dataset csv file , each row contains a "source" entry in english and a "translated" entry in Hindi/Marathi'''
        row = self.df.iloc[idx]
        if self.src_mode == "bert":
            '''if the encoder is a BERT architecture, get the BERT token ids and mask for the input sentence'''
            src_ids, src_mask = self._encode_src_bert(row["source"])
        else:
            '''If using the Bi-LSTM encoder, prepare GloVe token IDs and attention mask'''
            src_ids, src_mask = self._encode_src_glove(row["source"])
        '''get the BPE token ids for the translated sentence in the given row'''
        tgt_ids = self._encode_tgt(row["translated"])
        return {
            "src_ids": src_ids, 
            "src_attention_mask": src_mask, 
            "tgt_ids": tgt_ids
        }

'''
batch
│
├── sample 1
├── sample 2
├── sample 3
...
└── sample 32

i.e, batch is a list of samples returned by __getitem__(), one sample per dataset example
'''
def nmt_collate_fn(batch):
    '''Pad all source/target sequences and masks in the batch to a common length for batch training'''
    src_ids = pad_sequence([b["src_ids"] for b in batch], batch_first=True, padding_value=PAD_IDX)
    tgt_ids = pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=PAD_IDX)
    src_mask = pad_sequence([b["src_attention_mask"] for b in batch], batch_first=True, padding_value=0)
    
    '''Decoder input sequence (remove the last token)'''
    tgt_inputs = tgt_ids[:, :-1]  

    '''tgt_inputs are fed into decoder .. and loss will be based on o/p logits and tgt_labels'''
    
    '''Ground-truth target sequence (remove the first <SOS> token)'''
    tgt_labels = tgt_ids[:, 1:]   
    '''Create a mask indicating valid target tokens (ignore padding positions)'''
    tgt_mask = (tgt_labels != PAD_IDX).long()
    
    return {
        "src_ids": src_ids,
        "src_attention_mask": src_mask,
        "tgt_inputs": tgt_inputs, 
        "tgt_labels": tgt_labels,
        "tgt_attention_mask": tgt_mask,
    }
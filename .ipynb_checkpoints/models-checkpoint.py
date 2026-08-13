import torch
print(torch.__version__)
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

''' Bahdanau Attention Encoder-Decoder architecture '''
class Attention(nn.Module):
    '''dec_hidden_dim : 512 , final layer enc_hidden_dim : 512'''
    def __init__(self, dec_hidden_dim, enc_hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        self.U_a = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.v_a = nn.Linear(dec_hidden_dim, 1)
    def forward(self, dec_hidden, enc_states, src_mask):
        '''Previous hidden decoder state 'dec_hideen' : (B,512) and W : (512,512) ie, q : (B,1,512) unsqueezed!'''
        q = self.W_a(dec_hidden).unsqueeze(1)  
        '''Final encoder layer hidden states 'enc_states' : (B,T,512) and U : (512,512) ie, k : (B,T,512)'''
        k = self.U_a(enc_states) 
        '''energy dimension : (B,T,512)'''
        energy = torch.tanh(q + k) 
        '''score dimension : (B,T) squeezed! , scores for context vector @t for all encoder hidden states'''
        scores = self.v_a(energy).squeeze(2)
        '''masking the score of last '<pad>' token if the 'true' length of sequence in the batch < T'''
        scores = scores.masked_fill(src_mask == 0, float('-inf'))
        '''Softmax on attention scores'''
        attn_weights = F.softmax(scores, dim=1)
        '''Final context vector : (B,1,512) unsqueezed! to be fed into the decoder LSTM for predicting next logit and next decoder hidden state'''
        context = torch.bmm(attn_weights.unsqueeze(1), enc_states) 
        return context

class BaseSeq2Seq(nn.Module):
    def forward(self, src_ids, src_mask, tgt_inputs, tgt_mask):
        '''src_ids : (B,T) contains the token ids , token ids are used to get the token embedding from embedding matrix 
        [
         [93,34,23,45,67,45,3],
         [45,56,56,34,__,__,__],
         [34,65,39,__,__,__,__]
        ]                       ; where __ is <pad> 
        src_masks : (B,T) with value 0 for <pad> else 1 
        '''
        '''tgt_inputs : (B,T) are used for teaching forcing the decoder'''
        B, T = tgt_inputs.shape
        '''encoder_out is (out,h,c) where out : (B,T,512) , h : (num_layers*2,B,256) , c : (num_layers*2,B,256) '''
        encoder_out = self.encode(src_ids, src_mask)
        '''hidden is (h,c) where; h,c : (num_layers,B,512)'''
        hidden = self.init_decoder_hidden(encoder_out)
        '''logit_slices stores the logit of all time_stamps where logits are calculated iteratively over time'''
        logit_slices = []
        for t in range(T):
            '''token : teach force input tgt_input for time stamp @t'''
            token = tgt_inputs[:, t]
            '''step_logits : (B,10000) and new_hidden = (new_h,new_c) : (num_layers,B,512)'''
            step_logit, hidden = self.decode_step(token, hidden, encoder_out, src_mask)
            '''append the logit for the current time stamp @t'''
            logit_slices.append(step_logit.unsqueeze(1))
            
        logits = torch.cat(logit_slices, dim=1)
        return logits

class Model1_GloVe(BaseSeq2Seq):
    def __init__(self, src_vocab_matrix, tgt_vocab_size, dec_embed_dim=256,
                 enc_hidden_dim=256, dec_hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        '''src_embed is used to get the token embedding for the token_id'''
        self.src_embed = nn.Embedding.from_pretrained(src_vocab_matrix, freeze=False)
        '''enc_embed_dim = 768 (src_vocab_matrix.size(1)) , enc_hidden_dim = 256'''
        self.enc_lstm = nn.LSTM(src_vocab_matrix.size(1), enc_hidden_dim, num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        '''tgt_embed is used to get the token embedding (of dimension 256) for the tgt_input '''
        self.tgt_embed = nn.Embedding(tgt_vocab_size, dec_embed_dim)
        '''dec_embed_dim = 256 , dec_hidden_state = 512'''
        self.dec_lstm = nn.LSTM(dec_embed_dim, dec_hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        '''fc_out is the NN to get the decoder logits from decoder hidden states (B,512) -> (B,10000) ; where 10000 = tgt_vocab_size (say!)'''
        self.fc_out = nn.Linear(dec_hidden_dim, tgt_vocab_size)  
        self.dropout = nn.Dropout(dropout)
        '''encoder bi-LSTM stack hidden state dimension = 256 in one direction , final hidden state will be forward + backward = 512'''
        self.enc_hidden_dim = enc_hidden_dim
        '''decoder LSTM stack hidden state dimension = 512 , since h : (num_layers , B , 512) from hidden is passed as initial hidden state into decoder stack'''
        self.dec_hidden_dim = dec_hidden_dim
        '''num_layer = 2 , the number of LSTM layers in encoder/decoder stack'''
        self.num_layers = num_layers
        
    def forward(self, src_ids, src_mask, tgt_inputs, tgt_mask):
        '''encoder_out is (out,h,c) where out : (B,T,512) , h : (num_layers*2,B,256) , c : (num_layers*2,B,256) '''
        encoder_out = self.encode(src_ids, src_mask)
        '''hidden is (h,c) where; h,c : (num_layers,B,512)'''
        hidden = self.init_decoder_hidden(encoder_out) 
        '''tgt_inputs : (B,T) exist only for teacher forcing during training. During inference, the decoder feeds its own predictions back as inputs.'''
        '''embeds : (B,T,256) are the tgt_inputs embeddings'''
        embeds = self.dropout(self.tgt_embed(tgt_inputs))
        '''lstm_out : (B,T,512) are the final hidden states of decoder layers at all time stamps'''
        lstm_out, _ = self.dec_lstm(embeds, hidden)
        '''lstm_out are then passed into a neural net to get the logits : (B,T,10000) ; 100000 = tgt_vocab_size (say!)'''
        logits = self.fc_out(self.dropout(lstm_out))
        return logits
        
    '''src_ids : (B,T) , src_mask : (B,T)'''
    def encode(self, src_ids, src_mask):
        '''Get the token embeddings from the embedding matrix to get 'embed' : (B,T,768)'''
        embed = self.dropout(self.src_embed(src_ids))  
        '''lengths tells us how many true tokens are there in each sequence other than <pad>
        [
         [93,34,23,45,67,45,3],
         [45,56,56,34,__,__,__],
         [34,65,39,__,__,__,__]
        ]  
        for this lengths will be [7,4,3] 
        '''
        lengths = src_mask.sum(dim=1).cpu().clamp(min=1)
        '''based on lenghts, embed is the packed into [14,768] to avoid <pad> input embeddings being fed into the encoder'''
        packed = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        '''packed is then fed into the encoder bi-LSTM stacked layers to get final hidden states
           packed_out : (14,512) packed final layer hidden states of encoder LSTM stacks 
           h,c : hidden and cell states of each layer both in forward and backward direction (num_layers*2,B,256) 
        '''
        packed_out, (h, c) = self.enc_lstm(packed)
        '''out is unpacked packed_out : (B,T,512)'''
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src_ids.size(1))
        return (out, h, c)
        
    def init_decoder_hidden(self, encoder_out):
        '''encoder_out is (out,h,c) where out : (B,T,512) , h : (num_layers*2,B,256) , c : (num_layers*2,B,256) '''
        _, h, c = encoder_out
        '''L is num_layers'''
        L = self.num_layers
        '''B is batch size'''
        B = h.size(1) 
        '''H is 256'''
        H = self.enc_hidden_dim
        '''so, h,c :(num_layers*2,B,256) = (L*2,B,H)''' 
        def concat_directions(state):
            state = state.reshape(L, 2, B, H)
            '''gives one combined hidden state per encoder layer : (L,B,2*H) = (num_layers,B,512) , WHERE 2*H is (frontLSTM + backLSTM)'''
            return torch.cat([state[:, 0], state[:, 1]], dim=-1)
        '''(num_layers,B,512)'''
        return (concat_directions(h), concat_directions(c)) 
        
    def decode_step(self, token, hidden, encoder_out, src_mask):
        '''token : teach force input tgt_input for time stamp @t
           hidden is (h,c) where; h,c : (num_layers,B,512)
           encoder_out is (out,h,c) where out : (B,T,512) , h : (num_layers*2,B,256) , c : (num_layers*2,B,256)
           src_masks : (B,T) with value 0 for <pad> else 1 
        '''
        '''embed : (B,256) = (B,1,256) unsqueezed!'''
        embed = self.dropout(self.tgt_embed(token).unsqueeze(1))
        '''lstm_out : (B,1,512) , new_hidden = (new_h,new_c) : (num_layers,B,512)'''
        lstm_out, new_hidden = self.dec_lstm(embed, hidden)
        '''step_logits : (B,10000)'''
        step_logits = self.fc_out(self.dropout(lstm_out.squeeze(1)))
        '''hence, step_logits : (B,10000) and new_hidden = (new_h,new_c) : (num_layers,B,512) '''
        return step_logits, new_hidden

class Model2_GloVeAttn(Model1_GloVe):
    def __init__(self, src_vocab_matrix, tgt_vocab_size, dec_embed_dim=256,
                 enc_hidden_dim=256, dec_hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__(src_vocab_matrix, tgt_vocab_size, dec_embed_dim,
                         enc_hidden_dim, dec_hidden_dim, num_layers, dropout)
        '''Attention(512,512) , because final hidden states of encoder final layer dim = 512 i.e, 256(F) + 256(B)'''
        self.attention = Attention(dec_hidden_dim, 2*enc_hidden_dim)
        '''dec_lstm now take same hidden decoder state (num_layer,B,512) but input to decoder LSTM at each time stamp @t is now 'tgt_input(256) + context vector(512)' '''
        self.dec_lstm = nn.LSTM(dec_embed_dim + 2*enc_hidden_dim, dec_hidden_dim, num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
        '''fc_out is the NN to get the decoder logits from decoder hidden states (B,1024) -> (B,10000) ; where 10000 = tgt_vocab_size (say!)'''
        self.fc_out = nn.Linear(dec_hidden_dim + 2*enc_hidden_dim, tgt_vocab_size)

    def forward(self, src_ids, src_mask, tgt_inputs, tgt_mask):
        return BaseSeq2Seq.forward(self, src_ids, src_mask, tgt_inputs, tgt_mask)

    def decode_step(self, token, hidden, encoder_out, src_mask):
        '''encoder_out is (out,h,c) where out : (B,T,512) , h : (num_layers*2,B,256) , c : (num_layers*2,B,256) '''
        '''hidden is (h,c) where; h,c : (num_layers,B,512)'''
        '''token : teach force input tgt_input for time stamp @t'''
        '''src_masks : (B,T) with value 0 for <pad> else 1'''
        out, _, _ = encoder_out
        '''embed : (B,256) = (B,1,256) unsqueezed!'''
        embed = self.dropout(self.tgt_embed(token).unsqueeze(1)) 
        '''from the previous hidden state of decoder, take the last layer hidden state : (B,512) to calculate context vector for attention'''
        dec_h = hidden[0][-1] 
        '''Final context vector : (B,1,512) unsqueezed! to be fed into the decoder LSTM for predicting next logit and next decoder hidden state'''
        context = self.attention(dec_h, out, src_mask)
        '''embed : (B,1,256) , context : (B,1,512) , so input to decoder LSTM is (B,1,768) ,,, concat vertically'''
        lstm_input = torch.cat((embed, context), dim=2)
        '''lstm_out : (B,1,512) , new_hidden = (new_h,new_c) : (num_layers,B,512)'''
        lstm_out, new_hidden = self.dec_lstm(lstm_input, hidden)
        '''final input to feed forward NN to get the logit is concat horizontally {lstm_out (B,512) + [context(B,512) squeezed!]} = final_input (B,1024)'''
        final_input = torch.cat((lstm_out.squeeze(1), context.squeeze(1)), dim=1)
        '''final logit for current time stamp @t'''
        step_logits = self.fc_out(self.dropout(final_input))
        return step_logits, new_hidden

class Model3_FrozenBERT(BaseSeq2Seq):
    '''This model is a BERT encoder + 2 layer LSTM decoder'''
    def __init__(self, tgt_vocab_size, dec_embed_dim=256, dec_hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        for param in self.bert.parameters():
            param.requires_grad = False
        '''enc_hidden_dim = 768'''
        enc_hidden_dim = self.bert.config.hidden_size
        '''project final BERT output from 768 to 512'''
        self.bert_proj = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        '''h_proj and c_proj are used to get hidden = (h,c) from pooler_out to initiate decoder'''
        self.h_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        self.c_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        '''final encoder hidden states : 512 , decoder hidden state : 512'''
        self.attention = Attention(dec_hidden_dim, dec_hidden_dim)
        '''tgt_embed is used to get the token embedding (of dimension 256) for the tgt_input '''
        self.tgt_embed = nn.Embedding(tgt_vocab_size, dec_embed_dim)
        '''dec_lstm now take same hidden decoder state (num_layer,B,512) but input to decoder LSTM at each time stamp @t is now 'tgt_input(256) + context vector(512)' '''
        self.dec_lstm = nn.LSTM(dec_embed_dim + dec_hidden_dim, dec_hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        '''fc_out is the NN to get the decoder logits from decoder hidden states (B,1024) -> (B,10000) ; where 10000 = tgt_vocab_size (say!)'''
        self.fc_out = nn.Linear(dec_hidden_dim + dec_hidden_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        '''num_layer = 2 , the number of LSTM layers in encoder/decoder stack'''
        self.num_layers = num_layers
        '''decoder LSTM stack hidden state dimension = 512 , since h : (num_layers , B , 512) from hidden is passed as initial hidden state into decoder stack'''
        self.dec_hidden_dim = dec_hidden_dim

    def encode(self, src_ids, src_mask):
        outputs = self.bert(input_ids=src_ids, attention_mask=src_mask)
        '''outputs.last_hidden_states is final encoder hidden states and outputs.pooler_output is the final encoder hidden state for [CLS] token'''
        '''enc_states : (B,T,512) i.e, outputs.last_hidden_states : (B,T,768) is projected to enc_states : (B,T,512)'''
        enc_states = self.bert_proj(outputs.last_hidden_state)
        '''pooler_out : (B,512) i.e, outputs.pooler_output : (B,768) is projected to pooler_out : (B,512) 
           is then used to get hidden i.e, (h,c) : (num_layers,B,512) to initialize decoder stack'''
        pooler_out = self.bert_proj(outputs.pooler_output)
        return (enc_states, pooler_out)

    def init_decoder_hidden(self, encoder_out):
        '''(enc_states : (B,T,512), pooler_out : (B,512)) = encoder_out'''
        _, pooler_output = encoder_out
        '''get 'h' from pooler_output'''
        dec_h = self.h_proj(pooler_output)
        '''get 'c' from pooler_output'''
        dec_c = self.c_proj(pooler_output)
        '''change dimension of hidden = (h,c) : (B,512) to (num_layers,B,512)'''
        dec_h = dec_h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        dec_c = dec_c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        '''so, hidden = (dec_h,dec_c) : (num_layers,B,512)'''
        return (dec_h, dec_c)

    def decode_step(self, token, hidden, encoder_out, src_mask):
        '''enc_states : (B,T,512) final hidden states from BERT encoder'''
        enc_states, _ = encoder_out
        '''embed : (B,256) = (B,1,256) unsqueezed!'''
        embed = self.dropout(self.tgt_embed(token).unsqueeze(1))
        '''from the previous hidden state of decoder, take the last layer hidden state : (B,512) to calculate context vector for attention'''
        dec_h = hidden[0][-1]
        '''Final context vector : (B,1,512) unsqueezed! to be fed into the decoder LSTM for predicting next logit and next decoder hidden state'''
        context = self.attention(dec_h, enc_states, src_mask)
        '''lstm_input will be context vector : (B,1,512) + tgt_token_input (B,1,256) concatenated verically so (B,1,784)'''
        lstm_input = torch.cat((embed, context), dim=2)
        '''so, lstm_input : (B,1,784) and prev_hidden : (h,c) : (num_layers,B,512) are fed to dec_lstm'''
        '''lstm_out : (B,1,512) , new_hidden = (new_h,new_c) : (num_layers,B,512)'''
        lstm_out, new_hidden = self.dec_lstm(lstm_input, hidden)
        '''final input to feed forward NN to get the logit is concat horizontally {lstm_out (B,512) + [context(B,512) squeezed!]} = final_input (B,1024)'''
        final_input = torch.cat((lstm_out.squeeze(1), context.squeeze(1)), dim=1)
        '''final logit for current time stamp @t'''
        step_logits = self.fc_out(self.dropout(final_input))
        return step_logits, new_hidden

class Model4_TunedBERT(Model3_FrozenBERT):
    '''Same architecture as Model Three, but allow the transformer encoder to be fine-tuned.'''
    def __init__(self, tgt_vocab_size, dec_embed_dim=256, dec_hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__(tgt_vocab_size, dec_embed_dim, dec_hidden_dim, num_layers, dropout)
        for param in self.bert.parameters():
            param.requires_grad = True
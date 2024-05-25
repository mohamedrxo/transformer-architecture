#first we have to import the dipendencies mainly pytorch and math

import torch
from torch import Tensor
from torch.nn import functional as f
import math
import torch.nn as nn

#create the posisional encoding that will help identifiy the the index of the words

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# creating  the model that will be responsible for translationg from english to french 

class Translation_eng_fr(nn.Module):
    def __init__(self,d_model,vocab_size,tokenizer, num_layers=4 , num_heads=8, dropout=0.1):
        super(Translation_eng_fr, self).__init__()
        self.positionalencoding = PositionalEncoding(d_model,dropout)
        self.encoderlayer= nn.TransformerEncoderLayer(d_model, nhead=num_heads,dim_feedforward=d_model*4,dropout=dropout ,batch_first=True)
        self.transformer_encoder= nn.TransformerEncoder(self.encoderlayer, num_layers=num_layers)
        self.decoderlayer=nn.TransformerDecoderLayer(d_model=d_model,nhead=num_heads,batch_first=True)
        self.transformer_decoder= nn.TransformerDecoder(self.decoderlayer,num_layers=num_layers)
        self.Linear= nn.Linear(d_model,vocab_size)
        self.embadding=nn.Embedding(vocab_size,d_model)
        self.tokenizer=tokenizer
        self.drop= nn.Dropout(dropout)
        self.sqrt=math.sqrt(d_model)

    def create_padding_mask(self,seq):
            padding_mask = (seq != 50300).float()
            return padding_mask
    def generate_attention_mask(self,sequence_length):
        attention_mask = torch.triu(torch.ones((sequence_length, sequence_length)), diagonal=1)
        attention_mask *= float('-inf')
        attention_mask[attention_mask != attention_mask] = 0
        return attention_mask
    def encoder(self,src,src_):

         return self.transformer_encoder(src,src_key_padding_mask=self.create_padding_mask(src_))
    def decoder(self,tgt,out_encoder,tgt_):

         return self.transformer_decoder(tgt,out_encoder,tgt_mask=self.generate_attention_mask(tgt.shape[1]),tgt_key_padding_mask=self.create_padding_mask(tgt_))
    def embadding_(self,src):
         
         
         return  self.embadding(src)*self.sqrt

    def feedforward(self,x):
         return self.Linear(x)
    #the traing time
    def forward(self,src:Tensor,tgt:Tensor)->Tensor:
        
        tgt_=tgt
        
        src_=src

        src= self.embadding_(src)
        src=self.positionalencoding(src)
        tgt= self.embadding_(tgt)
        tgt=self.positionalencoding(tgt)
        out_encoder=self.encoder(src,src_)
        out_decoder=self.decoder(tgt,out_encoder,tgt_)

        out= self.feedforward(out_decoder)
        return out
    #the inference time
    def inference(self,src:Tensor)->Tensor:
        
        # 50258 is the start token and 50259 is the end token you can use the tiktoken librarie from openai to help you encode and decode the sentences
        
        tgt=torch.tensor([[50258]])
        
        
        src=torch.cat((torch.tensor([[50258]]), torch.tensor( [self.tokenizer.encode(src)]),torch.tensor([[50259]])),dim=1)
        
        src_=src
        src= self.embadding_(src)
        
        src=self.positionalencoding(src)
        
        out_encoder=self.encoder(src,src_)
        
        while True:
            tgt_=tgt
            tgts= self.embadding_(tgt)
            
            tgts=self.positionalencoding(tgts)
            
            
            logits=self.decoder(tgts,out_encoder,tgt_)
            
            logits = self.feedforward(logits)
            print(logits.shape)
            logits=logits[:,-1,:]
            print(logits.shape)
            
            logits = logits.view(-1)
            print(logits,logits.shape)
            probs= f.softmax(logits,dim=0)
            print(probs)
            
            
            prediction  = torch.argmax(probs)
            print("prediction ",prediction , probs[prediction])
            next_tgts=prediction.reshape(1,1)
            tgt=torch.cat((tgt,next_tgts),dim=1)
            

            print(eng_fr_tokenizer.decode(tgt.squeeze(0).tolist()))
            print(next_tgts)
            if next_tgts.item()==50259:
                print(eng_fr_tokenizer.decode(tgt.squeeze(0).tolist()))
                break
        return tgt

    
big_model=Translation_eng_fr(d_model=256,vocab_size=eng_fr_tokenizer.n_vocab,num_layers=3,tokenizer=eng_fr_tokenizer)

# printing the total number  parameters in the model

sum(p.numel() for p in big_model.parameters())

# finally note the you can use this model in GPU model py adding .cuda() to the end of each  object

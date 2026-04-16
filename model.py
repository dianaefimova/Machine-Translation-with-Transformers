#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:08:30 2025

@author: dianaefimova
"""

skip_training = False 

path = "data"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

# Set random seeds for all libraries
import random
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1) 

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load your data
en_df = pd.read_csv(os.path.join(path , 'small_vocab_en.csv'), header=None, usecols=[0])
fr_df = pd.read_csv(os.path.join(path, 'small_vocab_fr.csv'), header=None, usecols=[0])

english_sentences = en_df[0].values
french_sentences = fr_df[0].values

print(f'There are {len(english_sentences)} English sentences in data')
print(f'There are {len(french_sentences)} French sentences in data')
print('Here are some examples:')
e = [ 0, 1000, 3000]
for i in e:
    print(10*"-")
    print(english_sentences[i])
    print(french_sentences[i])
print(100*"_")

# Tokenize function 
def tokenize(sentences):

    filters = '.?!#$%&()*+,-/:;<=>@«»""[\\]^_`{|}~\t\n'

    tokenized_list=[]

    for sentence in sentences:
        sentence=sentence.lower()
        translation=str.maketrans('', '', filters)
        sentence=sentence.translate(translation)

        tokens=sentence.split()
 
        tokenized_list.append(tokens)
    
    return tokenized_list
    
# Tokenize English and French sentences
tokenized_en = tokenize(english_sentences)
tokenized_fr = tokenize(french_sentences)
for i in e:
    print(10*"-")
    print(tokenized_en[i])
    print(tokenized_fr[i])
    
    # Create vocabulary with special tokens
def build_vocab(tokenized_sentences):
    special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    tokens=[token for sentence in tokenized_sentences for token in sentence]
    
    vocab=Counter(tokens)
    
    vocab_list=special_tokens + list(vocab.keys())
    
    word2idx={word: idx for idx, word in enumerate(vocab_list)}
    
    idx2word = {idx: word for word, idx in word2idx.items()}

    
    return word2idx, idx2word

en_word2idx, en_idx2word = build_vocab(tokenized_en)
fr_word2idx, fr_idx2word = build_vocab(tokenized_fr)

print("Here are some examples from English dictionary: ")
print(100 * "-")

# Display first 10 words and their indices from en_word2idx
for i, (key, value) in enumerate(en_word2idx.items()):
    print(f'word: {key}, index: {value}')
    if i == 9:  
        break

print(10 * "_")

# Display first 10 indices and their words from en_idx2word
for i, (key, value) in enumerate(en_idx2word.items()):
    print(f'index: {key}, word: {value}')
    if i == 9: 
        break
    
# Dataset class with padding applied in __getitem__
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, seq_len=30):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.src_sentences)

    def pad_sequence(self, tokens, vocab, is_target=False):
        """
        Pads a sequence of tokens to the fixed length `seq_len`.
        Adds <SOS> at the start, <EOS> at the end, and pads with <PAD>.
        Trims if the sequence is longer than `seq_len`.
        """
        tokens = [vocab["<SOS>"]] + [vocab.get(token, vocab["<PAD>"]) for token in tokens]
        tokens.append(vocab["<EOS>"])  
        tokens = tokens[:self.seq_len]  
        tokens += [vocab["<PAD>"]] * (self.seq_len - len(tokens))  
        return tokens

    def __getitem__(self, idx):
        
        src_tokens = self.src_sentences[idx]
        tgt_tokens = self.tgt_sentences[idx]
        

        src_padded = self.pad_sequence(src_tokens, self.src_vocab, is_target=False)
        tgt_padded = self.pad_sequence(tgt_tokens, self.tgt_vocab, is_target=True)
        
        # Convert to tensors and move to device (GPU or CPU)
        src_item = torch.tensor(src_padded).to(device)
        tgt_item = torch.tensor(tgt_padded).to(device)
    
        return src_item, tgt_item

# Instantiate and test the dataset, so French is source language and English is target language.
dataset = TranslationDataset(tokenized_fr, tokenized_en, fr_word2idx, en_word2idx,  seq_len=10)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Test the DataLoader
for src_batch, tgt_batch in dataloader:
    print("Source batch:", src_batch)
    print(src_batch.size())
    print(10*"_")
    print("Target batch:", tgt_batch)
    print(tgt_batch.size())
    print(10*"_")
    break

embedding_size = 128
vsize_src = len(fr_word2idx)
vsize_tgt = len(en_word2idx)

dataset = TranslationDataset(tokenized_fr, tokenized_en, fr_word2idx, en_word2idx,  seq_len=10)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

src_batch, tgt_batch = next(iter(dataloader))

embedding_fr = nn.Embedding(num_embeddings=vsize_src, embedding_dim=embedding_size)
embedding_fr = embedding_fr.to(device)
output_embedding_fr = embedding_fr(src_batch.to(device))

# Define embedding for English vocabulary
embedding_en = nn.Embedding(num_embeddings=vsize_tgt, embedding_dim=embedding_size)
embedding_en = embedding_en.to(device)
output_embedding_en = embedding_en(tgt_batch.to(device))

print(10*"_")
print(src_batch.size())
print(output_embedding_fr.size())
print(10*"_")
print(tgt_batch.size())
print(output_embedding_en.size())

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        
        # Initialize a tensor to store positional encodings 
        pos_encoding = torch.zeros(max_len, embed_size)
        # Create a tensor for positions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term=torch.exp(torch.arange(0, embed_size, 2).float() *  (-math.log(10000.0) / embed_size))
        
        pos_encoding[:, 0::2] =torch.sin(position * div_term)
        
        pos_encoding[:, 1::2]=torch.cos(position * div_term)

        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to embeddings
        x = x * math.sqrt(self.embed_size)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        return x
    
# Applying encoding 
positional_encoding = PositionalEncoding(embedding_size, 512)
output_pe_fr = positional_encoding (output_embedding_fr)
output_pe_en = positional_encoding (output_embedding_en)
print(10*"_")
print(output_pe_fr.size())
print(output_pe_en.size())

class MySimpleTransformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, embed_size, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len=512):
        super(MySimpleTransformer, self).__init__()

        self.embed_size=embed_size
        self.src_embedding = nn.Embedding(vocab_size_src, embed_size)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, embed_size)
        self.positional_encoding=PositionalEncoding(embed_size, max_len)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            batch_first=False  
        )

        # Final linear layer to project transformer output to vocab size 
        self.fc_out = nn.Linear(embed_size, vocab_size_tgt)
        

    # Applying forward pass  
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        if tgt_mask != None:
            tgt_mask = tgt_mask.transpose(0,1)
            
        src_emb = self.src_embedding(src.transpose(0, 1))  # (batch, seq, embed)
        tgt_emb = self.tgt_embedding(tgt.transpose(0, 1))  # (batch, seq, embed)

        # Positional encoding
        src_emb = self.positional_encoding(src_emb)  # (batch, seq, embed)
        tgt_emb = self.positional_encoding(tgt_emb)  # (batch, seq, embed)

        src_emb = src_emb.transpose(0, 1)  # (seq, batch, embed)
        tgt_emb = tgt_emb.transpose(0, 1)  # (seq, batch, embed)

        # Forward pass through the Transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )

        output=self.fc_out(output)
        
        output = output.transpose(0, 1)
        return output
    
    def get_tgt_mask(self, tgt):
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        return tgt_mask
    
    def create_pad_mask(self, matrix):

        pad_token = 0
        return (matrix == pad_token)

def get_num_trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params} trainable parameters.')
    return num_params

class MyTransformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, embed_size, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len=512):
        super(MyTransformer, self).__init__() 
        self.embed_size=embed_size

        self.src_embedding=nn.Embedding(vocab_size_src, embed_size)
        self.tgt_embedding=nn.Embedding(vocab_size_tgt, embed_size)
        
        # Positional encoding
        self.positional_encoding=PositionalEncoding(embed_size, max_len)
        # Transformer block
        self.transformer=nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            batch_first=False
            )
        # Final linear layer to project transformer output to vocab size
        self.fc_out=nn.Linear(embed_size, vocab_size_tgt)
        

    
    def encode(self, src, src_padding_mask):

        src = src.transpose(0, 1)
        src_emb = self.src_embedding(src.transpose(0, 1))
        src_emb = self.positional_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)
        encoded = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask
        )
          
        return encoded

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask):
        tgt = tgt.transpose(0, 1)
        if tgt_mask != None:
            tgt_mask = tgt_mask.transpose(0,1)
        tgt_emb = self.tgt_embedding(tgt.transpose(0, 1))
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)
        decoded = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
 
        return decoded

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):
        memory = self.encode(src, src_padding_mask)
        output_decoder = self.decode(tgt, memory, tgt_mask, tgt_padding_mask) 
        output = self.fc_out(output_decoder)
        
        output = output.transpose(0, 1)
        return  output_decoder, output 
    
    def get_tgt_mask(self, tgt):
        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        return tgt_mask
    
    def create_pad_mask(self, matrix):

        pad_token = 0
        return (matrix == pad_token)

bs = 256
dataset = TranslationDataset(tokenized_fr, tokenized_en, fr_word2idx, en_word2idx,  seq_len=7)
number_of_sentences = len(tokenized_fr)
train_size = int((0.8)*number_of_sentences)
test_size = int(number_of_sentences - train_size)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=True)

# Model
embedding_size = 240 
vsize_src = len(fr_word2idx) # 336 
vsize_tgt = len(en_word2idx) # 201
hdim = 512
model = MyTransformer(vsize_src, vsize_tgt, embedding_size, 6, hdim, 4, 4, max_len=256)
model = model.to(device)

# Training 
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
num_epochs = 15

if not skip_training:
    epoch_train_losses = []
    epoch_validation_losses = []
    for epoch in range(num_epochs):
        model = model.to(device)
        model.train()  # Set model to training mode
        total_loss = 0
        num_samples = 0
        for src_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            src=src_batch
            tgt_input=tgt_batch[:, :-1]
            tgt_expected=tgt_batch[:, 1:]
            
            src_padding_mask=model.create_pad_mask(src)
            tgt_padding_mask = model.create_pad_mask(tgt_input).float()
            tgt_mask = model.get_tgt_mask(tgt_input)

            _, output = model(
                src,
                tgt_input,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )        
            
            
            output = output.to(device)
            output = output.contiguous().view(-1, vsize_tgt)
            tgt_expected = tgt_expected.contiguous().view(-1)    
            
            loss = criterion(output, tgt_expected)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_samples += src_batch.shape[0]
    
        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_loss = round(epoch_train_loss, 4)
        epoch_train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss}")
        
        
        model.eval()
        validation_loss = 0
        num_samples = 0
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src=src_batch
                tgt_input=tgt_batch[:, :-1]
                tgt_expected = tgt_batch[:, 1:]
                src_padding_mask = model.create_pad_mask(src)
                tgt_padding_mask = model.create_pad_mask(tgt_input).float()
                tgt_mask = model.get_tgt_mask(tgt_input)
                
                _, output = model(
                    src,                  
                    tgt_input,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    tgt_mask=tgt_mask,
                )

                
                output = output.to(device)
                output = output.contiguous().view(-1, vsize_tgt)
                tgt_expected = tgt_expected.contiguous().view(-1)  
                loss = criterion(output, tgt_expected)
                validation_loss += loss.item()
                num_samples += src_batch.shape[0]
                
            epoch_validation_loss = validation_loss / len(val_loader)
            epoch_validation_loss = round(epoch_validation_loss, 4)
            epoch_validation_losses.append(epoch_validation_loss)
            
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_validation_loss}")
        torch.save(model.state_dict(), 'model.pth')   
            
    print("Training completed.")
    torch.save(model.state_dict(), 'model.pth') 
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, epoch_train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, epoch_validation_losses, label='Validation Loss', color='red', marker='x')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    seq_len=10
    start_token=1
    end_token=2
    model.eval()  

# Convert src_sentence to tokenized integers in the vocabulary dictionary 
example_source_sentences = ["new jersey est parfois calme pendant l' automne.", "california est généralement calme en mars."]
example_tokenized = tokenize(example_source_sentences)
src_sentences = []
for ex in example_tokenized:
    ex_inds = []
    for t in ex:
        t_ind = fr_word2idx [t]
        ex_inds.append(t_ind)
    src_sentences.append(ex_inds)    

translated_sequences = []
for counter, src_sentence in enumerate(src_sentences):    
    # Convert source tokens to Tensor 
    src_tensor = torch.tensor(src_sentence, dtype=torch.long).unsqueeze(0).to(device)  
    
    src_padding_mask = model.create_pad_mask(src_tensor)
    memory = model.encode(src_tensor, src_padding_mask)

    
    # initialize the predicted tgt_tokens with start token
    tgt_tokens = torch.ones(1, 1).fill_(start_token).type(torch.long).to(device) #(1,1)

    for i in range(seq_len-1):
        tgt_mask = model.get_tgt_mask(tgt_tokens)
        decoded = model.decode(tgt_tokens, memory, tgt_mask, tgt_padding_mask=None)
        logits = model.fc_out(decoded)
        next_token_logits = logits[-1, 0, :]           
        _, next_token_id = torch.max(next_token_logits, dim=-1)


        next_tgt_item = next_token_id.view(1, 1) 
        tgt_tokens = torch.cat([tgt_tokens, next_tgt_item], dim=1)
        if next_token_id.item() == end_token:
            break

    
    translated_tokens = tgt_tokens.squeeze().tolist()
    translated_sentence = ' '.join ([en_idx2word[i] for i in translated_tokens[1:]])
    translated_sequences.append(translated_tokens)
    print("original_sentence:", example_source_sentences[counter])
    print("translated_sentence:", translated_sentence)
    print(10*'-')

np.save('translation.npy', np.array(translated_sequences, dtype=object))

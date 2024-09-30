import numpy as np
import pandas as pd
import tensorflow as tf
import json
import re
import random


vocab_size_1 = 10000
vocab_size_2 = 12000
batch_size = 64


def tf_lower_and_split_punct(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text


def tf_lower_and_split_punct_1(text):
    text = tf.strings.lower(text)
    text = tf.strings.strip(text)
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text


def decode_string(ints):
  strs = [chr(i) for i in ints]
  joined = ''.join(strs)
  return joined

def tokens_to_text(tokens, id_to_word):
    words = id_to_word(tokens)

    try:
       result = tf.strings.reduce_join(words, axis=-1, separator=" ").numpy()
    except:
      result = words.numpy()

    decoded = tf.strings.unicode_decode(result,'utf-8').numpy()
    decoded_sentence = decode_string(decoded)
    return decoded_sentence


vocab_size_1 = 10000
vocab_size_2 = 12000
units_1 = 128


class Encoder(tf.keras.layers.Layer):
    def __init__(self,fre_vocab_size = vocab_size_1,units = units_1):
        super(Encoder,self).__init__()
        
        self.vocab_size = fre_vocab_size
        self.units =units

        self.embedding = tf.keras.layers.Embedding(input_dim = fre_vocab_size,output_dim = units,mask_zero=True)
        self.lstm = tf.keras.layers.Bidirectional(merge_mode='sum',layer = tf.keras.layers.LSTM(units,return_sequences= True))

    def call(self,encoder_inputs):

        embedded_output = self.embedding(encoder_inputs)
        output = self.lstm(embedded_output)
        return output
    

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self,units=units_1):
        super().__init__()

        self.units =units

        self.mha = (tf.keras.layers.MultiHeadAttention(key_dim= units,num_heads=1))
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self,context,target):

        attn_output = self.mha(query = target,value = context)
        x = self.add([target,attn_output])
        x = self.layernorm(x)
        return x
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self,fre_vocab_size = vocab_size_1,units = units_1,tam_vocab_size = vocab_size_2):
        super(Decoder,self).__init__()

        self.fre_vocab_size = fre_vocab_size
        self.tam_vocab_size = tam_vocab_size
        self.units = units

        self.embedding = tf.keras.layers.Embedding(input_dim = tam_vocab_size,output_dim = units,mask_zero=True)
        self.pre_attention_rnn = tf.keras.layers.LSTM(units,return_sequences = True,return_state = True)
        self.attention = CrossAttention(units)
        self.post_attention_rnn = tf.keras.layers.LSTM(units = units,return_sequences=True)
        self.dense = tf.keras.layers.Dense(tam_vocab_size,activation = tf.nn.log_softmax)

    def call(self,context,target,state = None,return_state = False):

        embedding_output = self.embedding(target)
        x,state_h,state_c = self.pre_attention_rnn(embedding_output,initial_state=state)
        x = self.attention(context,x)
        x = self.post_attention_rnn(x)
        logits = self.dense(x)

        if return_state:
            return logits,[state_h,state_c]

        return logits
    

class Translator(tf.keras.Model):
    def __init__(self,fre_vocab_size =vocab_size_1,units = units_1,tam_vocab_size = vocab_size_2):
        super().__init__()
        self.encoder = Encoder(fre_vocab_size,units)
        self.decoder = Decoder(fre_vocab_size,units,tam_vocab_size)

    def call(self,inputs):
        context,target = inputs
        encoder_output = self.encoder(context)
        logits = self.decoder(encoder_output,target)

        return logits
    



def generate_next_token_0(decoder,context,next_token,done,state,eos_id = None):
    
    logits,state = decoder(context,next_token,state,return_state = True)

    logits = logits[:,-1,:]

    next_token = tf.argmax(logits,axis = -1)

    logits = tf.squeeze(logits)

    next_token = tf.squeeze(next_token)

    logit = logits[next_token].numpy()
    
    next_token = tf.reshape(next_token,shape  = (1,1))
    
    if next_token == eos_id :
        done = True
    return next_token,state,done,logit


def translate_0(model,text,max_length = 30,sos_id = None,eos_id = None,fre_vectorization = None,id_to_word = None):
    tokens,logits = [],[]
    #condition to convert only five letter words
    text = text.split(' ')
    text = [x.strip().strip(',').strip('?').strip('!').strip('"').strip('.') for x in text]
    text = [x for x in text if len(x)==5]
    # print('five _letter_words in input_text: ' ,len(text))
    if len(text) == 0:
        return 'The Text has no five letter words! Please try again'
    text = ' '.join(text)
    text = tf.convert_to_tensor(text)[tf.newaxis]
    context = fre_vectorization(text).to_tensor()
    context = model.encoder(context)
    state = [tf.zeros((1,units_1)),tf.zeros((1,units_1))]
    next_token = tf.fill((1,1),sos_id)
    done  = False

    for i in range(max_length):
        try:
            next_token,state,done,logit = generate_next_token_0(decoder = model.decoder,
                                                                context = context,
                                                                next_token = next_token,
                                                                done = done,
                                                                state = state,
                                                                eos_id = eos_id
                                                              )
        except:
            raise Exception('generate next token code issue')
        if done:
            break
        tokens.append(next_token)
        logits.append(logit)
    tokens = tf.concat(tokens,axis = -1)
    tokens = tf.squeeze(tokens)
    
    translation =tokens_to_text(tokens,id_to_word)

    return translation,
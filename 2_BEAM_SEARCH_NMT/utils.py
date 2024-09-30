import numpy as np
import pandas as pd
import tensorflow as tf
import random
import string
import re
import json
import copy

def tf_lower_and_split_punct(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text

def tokens_to_text(tokens, id_to_word):
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    return result

vocab_size_1 = 12000
units_1 = 128

class Encoder(tf.keras.layers.Layer):
    def __init__(self,vocab_size = vocab_size_1,units = units_1):
        super(Encoder,self).__init__()
        
        self.vocab_size = vocab_size
        self.units =units

        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size,output_dim = units,input_shape = (None,),mask_zero=True)
        self.lstm = tf.keras.layers.Bidirectional(merge_mode='sum',layer = tf.keras.layers.LSTM(units,return_sequences= True))

    def call(self,encoder_inputs):

        embedded_output = self.embedding(encoder_inputs)
        output = self.lstm(embedded_output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "units": self.units
        })
        return config
    

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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self,vocab_size = vocab_size_1,units = units_1):
        super(Decoder,self).__init__()

        self.vocab_size = vocab_size
        self.units = units

        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size,output_dim = units,mask_zero=True)
        self.pre_attention_rnn = tf.keras.layers.LSTM(units,return_sequences = True,return_state = True)
        self.attention = CrossAttention(units)
        self.post_attention_rnn = tf.keras.layers.LSTM(units = units,return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size,activation = tf.nn.log_softmax)

    def call(self,context,target,state = None,return_state = False):

        embedding_output = self.embedding(target)
        x,state_h,state_c = self.pre_attention_rnn(embedding_output,initial_state=state)
        x = self.attention(context,x)
        x = self.post_attention_rnn(x)
        logits = self.dense(x)

        if return_state:
            return logits,[state_h,state_c]

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "units": self.units
        })
        return config


class Translator(tf.keras.Model):
    def __init__(self,vocab_size =vocab_size_1,units = units_1):
        super().__init__()
        self.encoder = Encoder(vocab_size,units)
        self.decoder = Decoder(vocab_size,units)

    def call(self,inputs):
        context,target = inputs
        encoder_output = self.encoder(context)
        logits = self.decoder(encoder_output,target)

        return logits


english_vectorization =tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    output_mode = 'int',
    ragged=True,
    max_tokens=vocab_size_1,
    # output_sequence_length = 20
)

spanish_vectorization =tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    output_mode = 'int',
    ragged = True,
    max_tokens=vocab_size_1,
    # output_sequence_length=21
)


def generate_next_token(decoder,context,next_token,state,beam_width):
    
    logits,state = decoder(context,next_token,state,return_state = True)

    logits = logits[:,-1,:]

    next_logits , next_tokens = tf.nn.top_k(logits,k = beam_width)

    next_logits = tf.squeeze(next_logits).numpy()

    next_tokens = tf.squeeze(next_tokens).numpy()
    if beam_width == 1:
        return [next_tokens],state,[next_logits]

    return next_tokens,state,next_logits



def translate(model,text,max_length = 30,beam_width= 1,english_vectorizer = None,eos_id = None,sos_id =None,id_to_word = None):

    text = tf.convert_to_tensor(text)[tf.newaxis]

    context = english_vectorizer(text).to_tensor()

    context = model.encoder(context)
    state = [tf.zeros((1,units_1)),tf.zeros((1,units_1))]

    end_token = tf.fill((1,1),eos_id)

    done  = False

    sequences = [[[sos_id.numpy()],0.0,state]]

    final_sequences = []
    k = beam_width
    for i in range(max_length):

        if len(final_sequences)<k:

            if len(sequences)>k:
                sequences.sort(key = lambda x: x[1],reverse = True)
                sequences = sequences[:k]
            
            pre_sequences = []
            for i in range(len(sequences)):

                cur_sequence = sequences[i]

                cur_token = tf.cast(tf.fill((1,1),cur_sequence[0][-1]),end_token.dtype)

                cur_state = cur_sequence[2]

                if cur_token == end_token:
                    final_sequences.append(copy.deepcopy(cur_sequence))
                    continue

                next_tokens,state,next_logits = generate_next_token(decoder = model.decoder,
                                                                        context = context,
                                                                        next_token = cur_token,
                                                                        state = cur_state,
                                                                        beam_width = k)
                
    
                my_sequences = [copy.deepcopy(cur_sequence) for x in range(k)]

                for i in range(len(my_sequences)):
                    my_sequences[i][0].append(next_tokens[i])

                    my_sequences[i][1]+=next_logits[i]

                    my_sequences[i][2] = state

                pre_sequences+=my_sequences
                

            sequences = pre_sequences

    def cleaning(list_sequences):
        my_list = []
        list_sequences.sort(key = lambda x: x[1],reverse = True)
        
        if len(list_sequences)>k:
                list_sequences = list_sequences[:k]
        for sequence in list_sequences:
            my_tokens = sequence[0]
            score = sequence[1]
            translation = tokens_to_text(my_tokens,id_to_word)
            translation = translation.numpy().decode()
            my_list.append([translation,f'score: {round(score,3)}'])
        return my_list
    return cleaning(final_sequences)
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import re
import copy
from collections import Counter

# Lowercase, trim, and remove non-letter characters
def normalizeEng(s):
    s = str(s).lower().strip()
    s = re.sub(r"([.!?|,])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?.,]+", r" ", s)
    s = ' '.join(s.split())
    return s

def normalizeHin(s):
    s = str(s).lower().strip()
    s = re.sub(r"([.!?|,])", r" \1", s)
    s = re.sub(r"([\u0964])", r" \1", s)
    s = re.sub(r'[^\u0900-\u0965\u0970-\u097F!?.,]+',r" ",s)
    s = ' '.join(s.split())
    return s



def tf_lower_and_remove_punct(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?,!]", "")
    text = tf.strings.regex_replace(text, "[.?,!]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text


def tf_lower_and_remove_punct_1(text):
    text = tf.strings.regex_replace(text,'[^ \u0900-\u0965\u0970-\u097F?,!]','')
    text = tf.strings.regex_replace(text, "[\u0964?,!]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text


#function to create vocab using dataframe and a column
def vocab_forming(df,column):
    vocab = df[column].to_list()
    vocab = [i for j in vocab for i in j.split()]
    vocab = list(Counter(vocab).items())
    vocab = sorted(vocab,key = lambda x:x[1],reverse = True)
    vocab = [k for k,v in vocab]
    return vocab

#positional encoding
def positional_encoding(positions,d_model):
    position = np.arange(positions)[:,np.newaxis]
    k = np.arange(d_model)[np.newaxis,:]
    i = k//2
    angle_rates = 1/(np.power(10000,(2*i)/np.float32(d_model)))
    angle_rads = position*angle_rates
#     print('looks of anglerads,: ',angle_rads.shape)
    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
#     print('new axis angle_rads: ',pos_encoding.shape)
    return tf.cast(pos_encoding,dtype = tf.float32)

#PositionalEmbedding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,vocab_size,d_model,mask_zero=True):
        super().__init__()
        self.d_model = d_model
        self.mask_zero= mask_zero
        self.token_embedding = tf.keras.layers.Embedding(input_dim = vocab_size,output_dim= d_model,mask_zero=mask_zero)
        self.pos_encoding = positional_encoding(128,d_model)
    
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.token_embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[:, :length, :]
        return x
    def compute_mask(self,inputs,mask =None):
        if self.mask_zero:
            return tf.not_equal(inputs,0)
        else:
            return None

def FullyConnected(embedding_dim,dense_dim):
    feedforward = tf.keras.Sequential([
        tf.keras.layers.Dense(dense_dim,activation='relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])
    return feedforward

#Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,embedding_dim,num_heads,dense_dim,dropout_rate =0.1,layernorm_eps =1e-6,**kwargs):
        super().__init__(**kwargs)

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim = embedding_dim,
            dropout = dropout_rate
        )

        self.ffn = FullyConnected(embedding_dim=embedding_dim,dense_dim=dense_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)
        self.supports_masking = True
    def call(self,x,mask=None):
        # print('current_mask is: ',mask)
        if mask is not None:
            padding_mask = tf.cast(mask[:,None,:],dtype = tf.int32)
        else:
            padding_mask = None
#         print('current_padding_mask: ',padding_mask)
        attn_output = self.mha( x, x, x,attention_mask = padding_mask)
        layernorm1_output = self.layernorm1(x + attn_output)
        feedforward_output = self.ffn(layernorm1_output)
        dropout_ffn_output = self.dropout_ffn(feedforward_output)
        encoder_layer_output = self.layernorm2(layernorm1_output + dropout_ffn_output)

        return encoder_layer_output
#Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self,embedding_dim,num_heads,dense_dim,num_layers,input_vocab_size,
                 dropout_rate =0.1,layernorm_eps =1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(input_vocab_size,self.embedding_dim)
        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        dense_dim=dense_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.supports_masking =True
    def call(self,x,mask =None):
        # print('current mask: ',mask)
        x = self.embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x

#DecoderLayer   
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,embedding_dim,num_heads,dense_dim,dropout_rate =0.1,layernorm_eps =1e-6):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(key_dim=embedding_dim,num_heads = num_heads,dropout=dropout_rate)
        self.mha2 = tf.keras.layers.MultiHeadAttention(key_dim = embedding_dim,num_heads = num_heads,dropout =dropout_rate)
        self.ffn = FullyConnected(embedding_dim=embedding_dim,dense_dim=dense_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.supports_masking = True
    def call(self,x,enc_output,mask=None):
        casual_mask = self.get_casual_attention_mask(x)
        if mask is not None:
            padding_mask = tf.cast(mask[:,None,:],dtype = tf.int32)
#             padding_mask = tf.minimum(padding_mask,casual_mask)
        else:
            padding_mask = None

        attn_out1 ,attn_scores = self.mha1(x,x,x,attention_mask=casual_mask,
                                           return_attention_scores = True)
        self.last_attn_scores = attn_scores
        Q = self.layernorm1(x+attn_out1)

        attn_out2 = self.mha2(query=Q,key=enc_output,value=enc_output,
                             attention_mask = padding_mask)
        attn_out2 = self.layernorm2(Q+attn_out2)

        ffn_output = self.ffn(attn_out2)
        ffn_output = self.dropout(ffn_output)
        decoder_output = self.layernorm3(attn_out2+ffn_output)
        
        return decoder_output

    def get_casual_attention_mask(self,x):
        input_shape = tf.shape(x)
        batch_size,sequence_length = input_shape[0],input_shape[1]
        i = tf.range(sequence_length)[:,None]
        j = tf.range(sequence_length)
        mask = tf.cast(i>=j,tf.int32)
        mask = tf.reshape(mask,(1,input_shape[1],input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size,-1),
                tf.convert_to_tensor([1,1]),
            ],
            axis =0,
        )
        return tf.tile(mask,mult)
    
#Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self,embedding_dim,num_heads,dense_dim,num_layers,output_vocab_size,dropout_rate=0.1,layernorm_eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(output_vocab_size,self.embedding_dim)
        self.dec_layer = [DecoderLayer(embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               dense_dim=dense_dim,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self,x,enc_output):
        attention_weights = {}
        x = self.embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layer[i](x,enc_output)
        #update attention_weights dictionary with the attention weights of block 1 and block 2
            # attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
            # attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
        return x
        
#Transformer
class Transformer(tf.keras.Model):
    def __init__(self, embedding_dim, num_heads,dense_dim, num_layers,input_vocab_size, 
               output_vocab_size, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               dense_dim=dense_dim,
                               input_vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               dense_dim=dense_dim,
                               output_vocab_size=output_vocab_size, 
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = tf.keras.layers.Dense(output_vocab_size, activation=tf.nn.log_softmax)
    
    def call(self, inputs):
        # print(type(inputs))
        input_sentence,output_sentence = inputs
        # print('inputs okay')
        enc_output = self.encoder(input_sentence)
        # print('enc_output_generated')
        
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
        dec_output = self.decoder(output_sentence,enc_output)
        # print('dec_output_generated')
        
        # pass decoder output through a linear layer and softmax (~1 line)
        logits = self.final_layer(dec_output)
        ### END CODE HERE ###
        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
        # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits

#optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
def masked_loss(y_true, y_pred):
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    # Check which elements of y_true are padding
    mask = tf.cast(y_true != 0, loss.dtype)
    
    loss *= mask
    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    match*= mask

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

def generate_next_token(model,context,target,current_index,beam_width):
    
    logits = model([context,target])

    logits = logits[:,current_index,:]

    next_logits , next_tokens = tf.nn.top_k(logits,k = beam_width)

    next_logits = tf.squeeze(next_logits).numpy()

    next_tokens = tf.squeeze(next_tokens).numpy()
    if beam_width == 1:
        return [next_tokens],[next_logits]

    return next_tokens,next_logits

def beam_search_decoder(model,input_sentence,max_length = 25,beam_width=1,
                        english_vectorization=None,hindi_vectorization=None,hindi_index_lookup = None):
    tokenized_input_sentence = english_vectorization([input_sentence])
    end_token = "[EOS]"
    decoded_sentence = "[SOS]"

    sequences = [[decoded_sentence,0.0]]
    final_sequences = []

    for current_sequence_index in range(max_length):
        if len(final_sequences)<beam_width:
            if len(sequences)>beam_width:
                sequences.sort(key = lambda x:x[1],reverse = True)
                sequences = sequences[:beam_width]
            pre_sequences = []
            for i in range(len(sequences)):
                cur_sequence = sequences[i]

                if cur_sequence[0][-5::] == end_token:
                    final_sequences.append(copy.deepcopy(cur_sequence))
                    continue

                current_tokenized_target_sentence = hindi_vectorization([cur_sequence[0]])

                next_tokens,next_logits = generate_next_token(model = model,
                                                            context = tokenized_input_sentence,
                                                            target = current_tokenized_target_sentence,
                                                            current_index = current_sequence_index,
                                                            beam_width =beam_width)

                my_sequences = [copy.deepcopy(cur_sequence) for _ in range(beam_width)]
                for i in range(len(my_sequences)):
                    my_token =hindi_index_lookup[next_tokens[i]]
                    my_sequences[i][0] += " " + my_token
                    my_sequences[i][1]+=next_logits[i]
                pre_sequences+=my_sequences

            sequences = pre_sequences
        else:
            break
    if len(final_sequences)==0:
        sequences = sequences.sort(key = lambda x:x[1],reverse = True)
        return sequences[0][0]
    final_sequences = sorted(final_sequences,key = lambda x:x[1],reverse = True)
    translation = final_sequences[0][0]
    try:
        translation = translation.rstrip('[EOS]')
    except:
        pass
    return translation.lstrip('[SOS]')           

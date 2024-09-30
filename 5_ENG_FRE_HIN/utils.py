import numpy as np
import pandas as pd
import tensorflow as tf
import json
import re
import random



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
    text = tf.strings.regex_replace(text,'[^ \u0900-\u097F,I?!]','')
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text


def tf_lower_and_split_punct_2(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-zçèœêâôïûùüàáōëîé.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text

def decode_sentence(input_sentence,transformer, eng_vectorization= None,l2_vectorization =None,l2_index_lookup=None,
                    max_decoded_sentence_length =None,):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[SOS]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = l2_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = tf.argmax(predictions[0, i, :]).numpy().item(0)
        sampled_token = l2_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[EOS]":
            break
    return decoded_sentence

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,embed_dim,dense_dim,num_heads,**kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim = embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dense_dim,activation = 'relu'),
                tf.keras.layers.Dense(embed_dim)
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self,inputs,mask = None):
        if mask is not None:
            padding_mask = tf.cast(mask[:,None,:],dtype = tf.int32)
        else:
            padding_mask = None
        attention_output = self.mha(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = padding_mask
        )
        proj_input = self.layernorm_1(attention_output+inputs)
        proj_output = self.dense_proj(proj_input)
        output = self.layernorm_2(proj_input + proj_output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim':self.embed_dim,
            'dense_dim':self.dense_dim,
            'num_heads': self.num_heads,
        })
        return config
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,sequence_length,vocab_size,embed_dim,**kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim  = embed_dim
        )

        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim = sequence_length,
            output_dim = embed_dim
        )

    def call(self,inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start = 0,limit = length,delta =1)
        embeded_tokens = self.token_embeddings(inputs)
        embeded_position = self.position_embeddings(positions)
        return embeded_tokens + embeded_position
    
    def compute_mask(self,inputs,mask =None):
        if mask is not None:
            return tf.not_equal(inputs,0)
        else:
            return None
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'embed_dim': self.embed_dim,
        })
        return config

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,embed_dim,latent_dim,num_heads,**kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=embed_dim)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads = num_heads,key_dim=embed_dim)

        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(latent_dim,activation = 'relu'),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self,inputs,encoder_outputs,mask=None):
        casual_mask = self.get_casual_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:,None,:],dtype = tf.int32)
            padding_mask = tf.minimum(padding_mask,casual_mask)
        else:
            padding_mask = None

        attention_output1 = self.mha1(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = casual_mask
        )

        out_1 = self.layernorm_1(inputs + attention_output1)

        attention_output2 = self.mha2(
            query = out_1,
            value = encoder_outputs,
            key = encoder_outputs,
            attention_mask = padding_mask,
        )

        out_2 = self.layernorm_2(out_1 + attention_output2)
        proj_output = self.dense_proj(out_2)
        output = self.layernorm_3(proj_output + out_2)

        return output
    
    def get_casual_attention_mask(self,inputs):
        input_shape = tf.shape(inputs)
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
    

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads
        })
        return config
def transformer_model(sequence_length,vocab_size):
    # define emmbedding dimensions, latent dimensions, and number of heads
    embed_dim = 100
    latent_dim = 256
    num_heads = 2

    #Encoder
    encoder_inputs = tf.keras.Input(shape = (None,), dtype = "int64", name = "encoder_inputs")

    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)

    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)

    encoder = tf.keras.Model(encoder_inputs, encoder_outputs, name = "encoder")

    #Decoder
    decoder_inputs = tf.keras.Input(shape = (None,), dtype = "int64", name = "decoder_inputs")
    encoder_seq_inputs = tf.keras.Input(shape = (None, embed_dim), name = "encoder_seq_inputs")

    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)

    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoder_seq_inputs)

    x = tf.keras.layers.Dropout(0.5)(x)

    decoder_outputs = tf.keras.layers.Dense(vocab_size, activation = tf.nn.log_softmax)(x)

    decoder = tf.keras.Model([decoder_inputs, encoder_seq_inputs], decoder_outputs, name = "decoder")

    # Define the final model
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    transformer = tf.keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name = "transformer"
    )
    return transformer

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import string
import re
import json
import copy

import tkinter as tk
from tkinter import ttk
from tkinter import *

from utils import (Translator,tf_lower_and_split_punct,translate)

vocab_size_1 = 12000
units_1 = 128

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

# Load and set the English vocabulary
with open('text_vectorization_files/english_vocab.json') as json_file:
    english_vocab = json.load(json_file)
    english_vectorization.set_vocabulary(english_vocab)

# Load and set the Spanish vocabulary
with open('text_vectorization_files/spanish_vocab.json') as json_file:
    spanish_vocab = json.load(json_file)
    spanish_vectorization.set_vocabulary(spanish_vocab)

#conveting words to tokens 
word_to_id = tf.keras.layers.StringLookup(
    vocabulary = spanish_vocab,
    mask_token = "",
    oov_token = '[UNK]'
)
#converting tokens to words
id_to_word = tf.keras.layers.StringLookup(
    vocabulary = spanish_vocab,
    mask_token = '',
    oov_token = '[UNK]',
    invert = True
)

#Getting start of sentence and end of sentence tokens
eos_id = word_to_id('[EOS]')
sos_id = word_to_id('[SOS]')

#Initializing translator model object and loading saved weights
translator = Translator(vocab_size_1,units_1)
translator.compile(optimizer = 'adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics = ['accuracy'])
inp = tf.random.uniform(shape = [64,25],minval=0,maxval=10000,dtype = tf.int32,seed = 42)
tar_in = tf.random.uniform(shape = [64,25],minval=0,maxval=10000,dtype = tf.int32,seed = 42)
logits = translator((inp,tar_in))
translator.load_weights('model_weights/english_to_spanish.weights.h5')

def translate_to_spanish(english_sentence,beam_width):
    spanish_sentence = translate(translator,english_sentence,beam_width = beam_width,english_vectorizer=english_vectorization,
                                 eos_id= eos_id,sos_id=sos_id,id_to_word=id_to_word)
    try:
        for i in range(len(spanish_sentence)):
            spanish_sentence[i][0] = spanish_sentence[i][0].lstrip("[SOS]").rstrip("[EOS]")
    except:
        pass
    print("Spanish translation: ", spanish_sentence)
      
    return spanish_sentence
text = "Do you want me to explain it again?"
original_translation = "¿Quieres que te lo explique de nuevo?"
print(f"input_sentence: {text}")
sample_translation = translate_to_spanish(text,beam_width=3)
print(f"original_translation: ",{original_translation})
#Handling the translate button
def handle_translate():
    english_sentence = text_input.get("1.0", "end-1c")
    beam_width = horizontal_slider.get()
    print(beam_width)
    translation = translate_to_spanish(english_sentence,beam_width)
    translation_output.delete("1.0", "end")
    for i in range(1,len(translation)+1):
        translation_output.insert(END, f"{i}: {translation[i-1]}" +'\n')


# Setting up the main window
root = tk.Tk()
root.title("English to Spanish Language Translator")
root.geometry("800x700")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

#Displaying Example_inputs to quick check the effectiveness of beam search (as vocabulary limit is only 12000 and model accuracy 
#                                                                                                                       is only 80%)
examples = tk.Label(input_frame, text="Some examples to check the effectiveness of beam search ",
                     font=(font_style, 10),justify= 'left')
examples.pack(anchor =W)
examples_text = tk.Text(input_frame,height =3,width = 50,font = (font_style,10))
examples_text.pack(anchor=W)
my_list =['I want you to return to your seat . (beam_width = 2)',
          'Tom is very rich and handsome (beam_width = 10)', 
          "For goodness' sake, it doesn't say that! (beam_width=10)"
          ]
for sentence in my_list:
    examples_text.insert(END,sentence+'\n')

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the english text to be translated", font=(font_style, font_size, 'bold'))
input_heading.pack()
# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()

#Heading for slider
beam_width_heading = tk.Label(input_frame,text = "Beam width slider",font=(font_style, font_size, 'bold') )
beam_width_heading.pack()

#Slider for beam_width
scale_int = IntVar(value =1 )
horizontal_slider = tk.Scale(root,from_ = 1,to =10,orient= 'horizontal',variable=scale_int)
horizontal_slider.pack()


# Submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)
# Heading for output
output_heading = tk.Label(output_frame, text="Spanish Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=15, width=80, font=(font_style, font_size))
translation_output.pack()


root.mainloop()
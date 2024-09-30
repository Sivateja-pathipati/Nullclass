import numpy as np
import pandas as pd
import tensorflow as tf
import json
import re
import random
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk
from tkinter import *


from utils import Translator
from utils import tf_lower_and_split_punct,tf_lower_and_split_punct_1,translate_0


vocab_size_1 = 10000
vocab_size_2 = 12000
units_1 = 128

# vectorization
fre_vectorization = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size_1,
    output_mode = "int",
    ragged = True,
    standardize=tf_lower_and_split_punct
)

tam_vectorization = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size_2,
    output_mode = "int",
    ragged = True,
    standardize=tf_lower_and_split_punct_1
)

#load and set the French vocabulary
with open('text_vectorization_files/fre_vocab.json') as json_file:
    fre_vocab = json.load(json_file)
    fre_vectorization.set_vocabulary(fre_vocab)

# Load and set the Tamil vocabulary
with open('text_vectorization_files/tam_vocab.json') as json_file:
    tam_vocab = json.load(json_file)
    tam_vectorization.set_vocabulary(tam_vocab)


word_to_id = tf.keras.layers.StringLookup(
    vocabulary = tam_vocab,
    mask_token = "",
    oov_token = '[UNK]'
)

id_to_word = tf.keras.layers.StringLookup(
    vocabulary = tam_vocab,
    mask_token = '',
    oov_token = '[UNK]',
    invert = True
)

#assigning ids to sos and eos tokens
sos_id = word_to_id('[SOS]')
eos_id = word_to_id('[EOS]')


#Initializing the translator and loading the weights
#Initializing the translator and loading the weights
translator = Translator(vocab_size_1,units_1,vocab_size_2)
translator.compile(optimizer = 'adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction = 'none'),
                  metrics = ['accuracy'])
inp = tf.random.uniform(shape = [64,26],minval=0,maxval=10000,dtype = tf.int32,seed = 42)
tar_in = tf.random.uniform(shape = [64,19],minval=0,maxval=10000,dtype = tf.int32,seed = 42)
logits = translator((inp,tar_in))
print(logits.shape)
translator.load_weights('model_weights/french_to_tamil.weights.h5')

#translation function to translate the sentence
def translate_to_tamil(french_sentence):
    translation = translate_0(translator,french_sentence,eos_id =eos_id,sos_id = sos_id,
                              fre_vectorization=fre_vectorization,id_to_word = id_to_word)
    translation = translation[0]
    translation = translation.replace(' ','   ')
    return translation
text = "Surtout, soyez patient."
print(f"input_sentence: {text}")
print(f"translation: {translate_to_tamil(text)}")

#Handling the translate button
def handle_translate():
    french_sentence = text_input.get("1.0", "end-1c")
    translation = translate_to_tamil(french_sentence)
    translation_output.delete("1.0", "end")
    translation_output.insert(END, f" {translation}" +'\n')


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
examples = tk.Label(input_frame, text="Some example french sentences and five letter words ",
                     font=(font_style, 10),justify= 'left')
examples.pack(anchor =W)
examples_text = tk.Text(input_frame,height =8,width = 50,font = (font_style,10))
examples_text.pack(anchor=W)
my_list = ["S'il vous plaît, allez-y.",
           "êtres",
           "Je pense que je deviens fou.",
           "Vous devriez arrêter de fumer car c'est malsain.",
           "autre bûche",
           " Je jouais de la flûte quand j'étais au lycée",
           ' cible quels huilé',
           "belle fille noirs était"]
for sentence in my_list:
    examples_text.insert(END,sentence+'\n')

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the french text to be translated", font=(font_style, font_size, 'bold'))
input_heading.pack()
# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()




# Submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)
# Heading for output
output_heading = tk.Label(output_frame, text="Tamil Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=15, width=80, font=(font_style, font_size))
translation_output.pack()


root.mainloop()
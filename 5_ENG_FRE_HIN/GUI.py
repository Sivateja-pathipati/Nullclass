#Importing requried packages
import numpy as np
import tensorflow as tf
import random
import string
import json
import re
import tkinter as tk
from tkinter import*
from tkinter import ttk

#Importing funcitons from utils file
from utils import tf_lower_and_split_punct,tf_lower_and_split_punct_1
from utils import tf_lower_and_split_punct_2,decode_sentence,transformer_model

# deifining configuration for vectorization
vocab_size = 15000
sequence_length1 = 50
sequence_length2 = 20
# vectorization of eng for eng_hin model
english_vectorization_hin = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length = sequence_length1,
    standardize=tf_lower_and_split_punct
)
# vectorization of hin for eng_hin model
hindi_vectorization = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length = sequence_length1+1,
    standardize=tf_lower_and_split_punct_1
)
# vectorization of eng for eng_fre model
english_vectorization_fre = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length = sequence_length2,
    standardize=tf_lower_and_split_punct
)
# vectorization of fre for eng_fre
french_vectorization = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length = sequence_length2+1,
    standardize=tf_lower_and_split_punct_2
)

##Load and set the English vocabulary for eng_hin model
with open('text_vectorization_files/english_vocab_for_eng_hin.json') as f:
    eng_vocab = json.load(f)
    english_vectorization_hin.set_vocabulary(eng_vocab)

# Load and set the Hindi vocabulary for eng_hin model
with open('text_vectorization_files/hindi_vocab_for_eng_hin.json') as f:
    hin_vocab = json.load(f)
    hindi_vectorization.set_vocabulary(hin_vocab)

#Load and set the English vocabulary for eng_fre model
with open('text_vectorization_files/english_vocab_for_eng_fre.json') as f:
    english_vocab = json.load(f)
    english_vectorization_fre.set_vocabulary(english_vocab)

# Load and set the French vocabulary for eng_fre model
with open('text_vectorization_files/french_vocab_for_eng_fre.json') as f:
    french_vocab = json.load(f)
    french_vectorization.set_vocabulary(french_vocab)

# Setting index lookup dictionary for later decoding 

hin_index_lookup = dict(zip(range(len(hin_vocab)), hin_vocab))
fre_index_lookup = dict(zip(range(len(french_vocab)),french_vocab))

# Loading the saved English-Hindi Model
transformer_hindi = transformer_model(vocab_size=vocab_size,sequence_length=sequence_length1)
transformer_hindi.load_weights("model_weights/english_hindi_model.weights.h5")

# Loading the saved English-French Model
transformer_french = transformer_model(vocab_size=vocab_size,sequence_length=sequence_length2)
transformer_french.load_weights("model_weights/english_french_model.weights.h5")


# Definign Translating to Hindi Function using utils decode_sentence function
def translate_to_hindi(english_sentence):
    hindi_sentence = decode_sentence(english_sentence,transformer_hindi,eng_vectorization=english_vectorization_hin,
                                     l2_vectorization=hindi_vectorization,l2_index_lookup=hin_index_lookup,
                                     max_decoded_sentence_length=sequence_length1)
    print("Hindi translation: ", hindi_sentence)
    
    return hindi_sentence.replace("[SOS]", "").replace("[EOS]", "")

# Defining Translating to French Function using utils decode_sentence function
def translate_to_french(english_sentence):
    french_sentence = decode_sentence(english_sentence,transformer_french,eng_vectorization=english_vectorization_fre,
                                     l2_vectorization=french_vectorization,l2_index_lookup=fre_index_lookup,
                                     max_decoded_sentence_length=sequence_length2)
    print("french translation: ", french_sentence)
    
    return french_sentence.replace("[SOS]", "").replace("[EOS]", "")

#checking wether model weights are working
translate_to_french("receptions")
translate_to_hindi("receptions")


# Setting up the main window
root = tk.Tk()
root.title("Language Translator")
root.geometry("800x700")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)



#Displaying Example 10 Letter Words
examples = tk.Label(input_frame, text="Some 10 letter words ",
                     font=(font_style, 12,'bold'),justify= 'left')
examples.pack(anchor =W)
examples_text = tk.Text(input_frame,height =8,width = 80,font = (font_style,10))
examples_text.pack(anchor=W)
my_list = ['collection deflection remissions articulate terrrorist catwalking celebrated',
           'bharatendu mujahideen mobilizing interrupts transpires classmates jahawarlal',
            'evangelist habituated ascendancy currencies enactments contracted', 
            'switchover vaisheshik assistance aparbrahma undetected applicants', 
            'mistresses suchindram recuperate prosecuted deforested glorifying', 
            'inoculated pleasantly thereafter revengeful humiliated vibheeshan parushuram kurukhatra',
            'portuguese electicity shrotyagya',]
for sentence in my_list:
    examples_text.insert(END,sentence+'\n')

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the text to be translated", font=(font_style, font_size, 'bold'))
input_heading.pack()
# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()

# Language selection
language_var = tk.StringVar()
language_label = tk.Label(root, text="Select the language to translate to", font=(font_style, font_size, 'bold'))
language_label.pack()
language_select = ttk.Combobox(root, textvariable=language_var, values=["French", "Hindi"], font=(font_style, font_size), state="readonly")
language_select.pack()


# Function to handle translation request based on selected language
def handle_translate():
    selected_language = language_var.get()
    english_sentence = text_input.get("1.0", "end-1c")\
    # Getting only 10 letter words in english sentence
    english_sentence = tf_lower_and_split_punct(english_sentence).numpy().decode()
    english_sentence = ' '.join([i for i in english_sentence.split(' ') if len(i)==10])
    print('10 letter words in english sentence: ',english_sentence)
    if selected_language == "French":
        translation = translate_to_french(english_sentence)
    elif selected_language == "Hindi":
        translation = translate_to_hindi(english_sentence)
        
    translation_output.delete("1.0", "end")
    translation_output.insert(END,'\n')
    translation_output.insert("end", f"{selected_language} translation: {translation}")


# Submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack(pady=10)



# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)
# Heading for output
output_heading = tk.Label(output_frame, text="Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=10, width=50, font=(font_style, 13,'bold'))
translation_output.pack()

# Running the application
root.mainloop()
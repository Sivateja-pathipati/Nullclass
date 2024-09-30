import numpy as np
import pandas as pd
import tensorflow as tf
import json
import re
import random
import copy
from collections import Counter
import datetime
import warnings
warnings.filterwarnings('ignore')

from utils import normalizeEng,tf_lower_and_remove_punct,tf_lower_and_remove_punct_1,beam_search_decoder
from utils import Transformer

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

import speech_recognition as sr
import whisper
import time

#Loading base model in whisper for later transcribing the results
base_model = whisper.load_model("base")

with open('Text_vectorization_file/english_vocab.json','r',encoding = 'utf-8') as f:
    english_vocab = json.load(f)

with open('Text_vectorization_file/hindi_vocab.json','r',encoding = 'utf-8') as f:
    hindi_vocab = json.load(f)

max_vocab_size =15000
sequence_length = 25

# Recreate the English & Hindi vectorization layers with basic configuration
english_vectorization = tf.keras.layers.TextVectorization(
    max_tokens = max_vocab_size,
    output_mode = 'int',
    output_sequence_length = sequence_length,
    standardize = tf_lower_and_remove_punct
)


hindi_vectorization = tf.keras.layers.TextVectorization(
    max_tokens = max_vocab_size,
    output_mode = 'int',
    output_sequence_length = sequence_length+1,
    standardize = tf_lower_and_remove_punct_1
)

english_vectorization.set_vocabulary(english_vocab)
hindi_vectorization.set_vocabulary(hindi_vocab)

# Initiating transformer object
embed_dim = 128
heads = 2
latent = 512
no_of_layers = 2
vocab_size_eng = len(english_vocab)
vocab_size_hin = len(hindi_vocab)

transformer1 = Transformer(embedding_dim=embed_dim,num_heads=heads,dense_dim=latent,
                          num_layers=no_of_layers,input_vocab_size=vocab_size_eng,
                           output_vocab_size=vocab_size_hin)

# Initializing transformer object with random input, it is a necessary step to set the loaded weights  
# if not initalized it produces falses results
tf.random.set_seed(42)
inp = tf.random.uniform(shape = [64,25],minval=0,maxval=10000,dtype = tf.int32,seed = 42)
tar_in = tf.random.uniform(shape = [64,25],minval=0,maxval=10000,dtype = tf.int32,seed = 42)
logits = transformer1((inp,tar_in))

# Note load the weights after initializing with random input
transformer1.load_weights('eng_hin.weights.h5')

# Translation function to translate the sentence
hindi_index_lookup = dict(zip(range(len(hindi_vocab)), hindi_vocab))
def translate_to_hindi(input_sentence):
    translation = beam_search_decoder(model =transformer1,input_sentence =input_sentence,beam_width=3,
                                      english_vectorization=english_vectorization,hindi_vectorization=hindi_vectorization,
                                      hindi_index_lookup=hindi_index_lookup)
    # translation = translation.replace(' ','   ')
    return translation

# Checking if translation is working correctly
text = "How are you"
print(f"input_sentence: {text}")
print(f"checking _translation: {translate_to_hindi(text)}")

# Code to check the oov words
english_vocab_set = set(english_vocab)
def check_sentence_qulaity(text):
    my_set = set(text.split())
    no_of_words = len(my_set)
    no_of_words_not_in_vocab = len(my_set-english_vocab_set)
    return f"There are {no_of_words_not_in_vocab} Out of Vocabulary Words in Input Sentence "

# Filtering words that are not starting with vowels
def m_o_filter(text):
    text = text.split()
    normal_len = len(text)
    words_start_with_vowel = False
    text = [i for i in text if not(i.startswith(('m','o')))]
    if len(text)<normal_len:
        words_start_with_vowel = True
    return " ".join(text),words_start_with_vowel


# Setting up the main window
root = tk.Tk()
root.title("English to Hindi Language Translator")
root.geometry("800x700")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)



def audio_transcriber():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold=3
    try: 
        with sr.Microphone() as source2:
            audio2 = recognizer.listen(source2,timeout=3)
            with open("myaudio.mp3", "wb") as f:
                f.write(audio2.get_wav_data())
        
        audio2 = whisper.load_audio("myaudio.mp3")
        results = base_model.transcribe(audio2,language = 'en')
        print(results['text'])
    except:
        record_label.config(text = "Unknown error: please record again")
    response = messagebox.askyesno('confirmation_box', f"Do you mean: {results['text']}")
    if response:
        text_input.delete("1.0", "end")
        text_input.insert(END,str(results['text']))
        try:
            handle_translate()
            mylistener.destroy()
        except:
            mylistener.destroy()
            audio_functionality()
            record_label.config(text ='Please record again')

    else:
        mylistener.destroy()
        audio_functionality()
        record_label.config(text ='Please record again')
        
        



def audio_functionality():
    global mylistener,record_label
    mylistener = Toplevel()
    mylistener.title('Audio to text converter')
    mylistener.geometry("400x300")
    recording_button = tk.Button(mylistener,text = "Start Recording",height =3,width =30,command = audio_transcriber)
    exit_button = tk.Button(mylistener,text = "EXIT", height =3,width = 20,command = mylistener.destroy)
    recording_button.grid(row = 5,column =0,columnspan=2,padx =10,pady = 20)
    exit_button.grid(row =5,column =2,columnspan=2)
    record_label = tk.Label(mylistener,text = "Hello",font=(font_style, font_size, 'bold'))
    record_label.grid(row =0,column = 0,columnspan=4)

    



audio_heading = tk.Label(input_frame, text="You can use the audio feature after 6pm", font=(font_style, font_size, 'bold'))

audio_button = tk.Button(input_frame,text = "Audio Listener",height = 3,width =20,command= audio_functionality,state=DISABLED)

operating_time = datetime.time(18,0,0,0)
current_time = datetime.datetime.now().time()
if current_time>=operating_time:
    audio_button.config(state = NORMAL)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the english text manually", font=(font_style, font_size, 'bold'))
# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))

audio_heading.grid(row=0,column=1,)
audio_button.grid(row=1,column =1,pady=10)
input_heading.grid(row=2,column =1)
text_input.grid(row=3,column=1)

#Handling the translate button
def handle_translate():
    submit_button.config(state = DISABLED)
    # Getting input from text box
    input_sentence = text_input.get("1.0", "end-1c")
    input_sentence = normalizeEng(input_sentence)

    # Checking out of vocabulary words
    sentence_quality = check_sentence_qulaity(input_sentence)
    
    # To filter words that start with consonants
    input_sentence,_ = m_o_filter(input_sentence)
        
    translation = translate_to_hindi(input_sentence)

    print("interpreted_input_sentence : ",input_sentence)
    print("           output_sentence : ",translation)  

    # posting the results in translation box
    translation_output.delete("1.0", "end")
    translation_output.insert(END,sentence_quality+'\n\n')
    translation_output.insert(END,f"interpreted_input_sentene: {input_sentence}"+'\n\n')
    translation_output.insert(END,f"Translation : {translation}" +'\n')
    # this label is to check result history and to perfectly show hindi words,
    #  the tranlsation output text box hindi_display is not good
    output_label = tk.Label(output_frame,text = translation,font=(font_style, font_size,))
    output_label.pack(anchor =W)




# Submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack(pady=10)
# Function to enable submit button
def enable():
    submit_button.config(state = NORMAL)

# Enable button to enable Translate button after each translation to avoid repetition
enable_button = ttk.Button(root,text = 'Enable the Translate button',command = enable)
enable_button.pack()

mylabel = tk.Label(root,text = "Note: words starting with M and O are not translated" , 
                   font=(font_style, 12, 'bold'),fg = 'black')
mylabel.pack()



# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)
# Heading for output
output_heading = tk.Label(output_frame, text="Hindi Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=7, width=80, font=(font_style, font_size),fg = 'black')
translation_output.pack()

#Result History
output_label = tk.Label(output_frame,text = 'Result History:',font=(font_style, font_size, 'bold'))
output_label.pack(anchor=W)


root.mainloop()

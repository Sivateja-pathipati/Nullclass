import tkinter as tk
from tkinter import ttk

import numpy as np
import tensorflow as tf
import json
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load  French model
model = load_model('english_to_french_model.h5')

#load Tokenizer
with open('tokenizers/english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)
    
with open('tokenizers/french_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)
    

#load max length
max_length =21
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    
    english_sentence = english_sentence.reshape((-1,max_length))

    french_sentence = model.predict(english_sentence)[0]
    
    french_sentence = [np.argmax(word) for word in french_sentence]

    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
    
    print("French translation: ", french_sentence)
    return french_sentence

def handle_translate():
    english_sentence = text_input.get("1.0", "end-1c")
    translation = translate_to_french(english_sentence)
    translation_output.delete("1.0", "end")
    translation_output.insert("end", f"french translation: {translation}")
# Setting up the main window
root = tk.Tk()
root.title("English to French Language Translator")
root.geometry("550x600")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the english text to be translated", font=(font_style, font_size, 'bold'))
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
output_heading = tk.Label(output_frame, text="Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=10, width=50, font=(font_style, font_size))
translation_output.pack()


root.mainloop()
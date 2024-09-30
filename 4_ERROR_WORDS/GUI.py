import random
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from tkinter import *

#Loading required functions from utils file
from utils import suggestions_1st_time,suggestions_2nd_time

#Our Vocabulary
with open('data/words_alpha.txt','r') as f:
    data =f.readlines()
    vocabulary = [x.strip('\n') for x in data]
vocabulary = set(vocabulary)


# Setting up the main window
root = tk.Tk()
root.title("SPELL CHECK & SUGGESTIONS APP")
root.geometry("500x300")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the english word for spell check",
                         font=(font_style, font_size, 'bold'))
input_heading.grid(row = 0,column = 0)

# Text input for English word
text_input = tk.Text(input_frame, height=1, width=15,font=(font_style, font_size))
text_input.grid(row = 1,column = 0,padx = (30,0),pady = 10,ipadx = 100)

#Heading for slider
suggestions_heading = tk.Label(input_frame,text = "No.of suggestions required",font=(font_style, font_size, 'bold'))
suggestions_heading.grid(row = 2,column = 0)

#Slider for Suggestions number
scale_int = IntVar(value =2 )
horizontal_slider = tk.Scale(input_frame,from_ = 2,to =7,orient= 'horizontal',variable=scale_int, )
horizontal_slider.grid(row = 3,column = 0,padx = (30,0),ipadx = 100)


#handling the check button
count = 1
def handle_check_button():
    global root,mylabel,count,suggestions1,word_1

    check_button.config(state = DISABLED)

    #Getting word and no.of.suggestions from user input
    word = text_input.get("1.0", "end-1c")
    word = word.lower().strip()
    n = horizontal_slider.get()
    mylabel.destroy()

    if word in vocabulary:
        
        #Desstroying the popup window if displayed
        try:
            pop.destroy()
        except:
            pass
        # Indicating the entered word is correct
        mylabel = tk.Label(root,text = 'Congrats your word exists in Vocabulary',font=(font_style, 14, 'bold'),fg = 'green')
        mylabel.pack()

        # destroying suggestions, word and count
        suggestions1 = None
        word_1 = None
        count = 1


    else:
        # refer popup function below for enterning the wrong word
        mylabel.destroy()
        my_popup(word,n)


# pop funciton acting as custom error box to implement assignment conidition
def my_popup(word,n):
    global pop,word_1,suggestions1,count,root,mylabel


    if count ==1:
        # To close the popup if it is open
        try: 
            pop.destroy()
        except:
            pass

        #Creating popup for 1st time
        pop = Toplevel(root)
        pop.title('ERROR')
        pop.geometry("250x300")
  
        # Indicating the entered word is wrong
        mylabel = tk.Label(root,text = 'You have entered the wrong word.', font=(font_style, 12, 'bold'),fg = 'Red')
        mylabel.pack()

        #Getting suggestions related to the word
        suggestions1,req_words = suggestions_1st_time(word,vocab_list=vocabulary)
        print(req_words[:n])


        #Displaying the  available suggestions
        if len(req_words)<=n:
            mylabel2 = tk.Label(pop,text = 'Not enough suggestions available: \n Suggestions:  ', font=(font_style, 12, 'bold'),fg = 'black')
            mylabel2.pack()
            str1 = ''
            for i in req_words:
                str1+= '\n' + i
            mylabel3 = tk.Label(pop,text = str1, font=(font_style, 12, 'bold'),fg = 'green')
            mylabel3.pack()

        else:
            mylabel2 = tk.Label(pop,text = 'Here are some suggestions: ', font=(font_style, 14, 'bold'),fg = 'black')
            mylabel2.pack()
            str1 = ''
            for i in range(n):
                str1+= '\n' + req_words[i]
            mylabel3 = tk.Label(pop,text = str1, font=(font_style, 12, 'bold'),fg = 'blue')
            mylabel3.pack()
            
        word_1 = word
        # Redefining the count value to keep track
        count =2
    
    else:
        # To close the popup if it is open
        try:
            pop.destroy()
        except: pass

        #Creating popup for 2nd time
        pop = Toplevel(root)
        pop.title('ERROR')
        pop.geometry("250x300")

        # Indicating the entered word is wrong
        mylabel.destroy()
        mylabel = tk.Label(root,text = 'You have entered the wrong word two times continuously.',font=(font_style, 12, 'bold'),fg = 'Red')
        mylabel.pack()

        #Getting suggestions related to the word
        suggestions = suggestions_2nd_time(word,suggestions1,vocab_list=vocabulary, n = n)
        print(suggestions[:n])

        # Displaying the wrong words entered so far
        mylabel4 = tk.Label(pop,text = 'Words you entered so far: ',font=(font_style, 12, 'bold'),fg = 'black')
        mylabel4.pack()
        mylabel5 = tk.Label(pop,text = word_1 + '\n' + word,font=(font_style, 12, 'bold'),fg = 'Red')
        mylabel5.pack()


        #Displaying the  available suggestions
        if len(suggestions)<n:
            mylabel2 = tk.Label(pop,text = 'Not enough suggestions available  \n Suggestions:  ', font=(font_style, 12, 'bold'),fg = 'black')
            mylabel2.pack()
            str1 = ''
            for i in suggestions:
                str1+= '\n' + i
            mylabel3 = tk.Label(pop,text = str1, font=(font_style, 12, 'bold'),fg = 'green')
            mylabel3.pack()
        else:
            mylabel2 = tk.Label(pop,text = 'Here are some suggestions: ', font=(font_style, 14, 'bold'),fg = 'black')
            mylabel2.pack()
            str1 = ''
            for i in range(n):
                str1+= '\n' + suggestions[i]
            mylabel3 = tk.Label(pop,text = str1, font=(font_style, 12, 'bold'),fg = 'blue')
            mylabel3.pack()

        # Assigning count value to initial value
        count = 1
        


# Submit button
check_button = ttk.Button(root, text="Check the word", command=handle_check_button,state = DISABLED)
check_button.pack()
def enable():
    check_button.config(state = NORMAL)

enable_button = ttk.Button(root,text = 'Enable the check the word button',command = enable)
enable_button.pack()
#Initial Note
note_label = tk.Label(root,text = 'please wait for suggestion results if your word is incorrect and long',font=(font_style, 10, 'italic'),fg = 'blue')
note_label.pack(pady = 10)

#Initial Display on window
mylabel = tk.Label(root,text = 'Thank you',font=(font_style, 12, 'bold'),fg = 'blue')
mylabel.pack()

root.mainloop()
import random

def process_word(word):
    word = word.lower().strip()
    word ="".join([i for i in word if i.isalpha()])
    return word

def removing_letter_ops(word):
   split_word = [(word[:i],word[i:]) for i in range(len(word)+1)]

   delete_letter = [L + R[1:] for (L,R) in split_word if R]
   return delete_letter

def switch_letter_ops(word):
    split_word = [(word[:i],word[i:]) for i in range(len(word)+1)]

    switch_letter = [L[:-1]+R[0]+L[-1]+R[1:] for L,R in split_word if (L) and (R)]
    return switch_letter

def replace_letter_ops(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    split_word = [(word[:i],word[i:]) for i in range(len(word)+1)]

    replace_letter = [L[:-1] + i + R[::] for L,R in split_word if L for i in letters]

    while word in replace_letter:
        replace_letter.remove(word)

    replace_letter = list(set(replace_letter))
    replace_letter = sorted(replace_letter)
    
    return replace_letter

def insert_letter_ops(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    split_word = [(word[:i],word[i:]) for i in range(len(word)+1)]

    insert_letter = [L + i +R for L,R in split_word for i in letters]
    return insert_letter

def edit_one_letter(word):
    word = word.lower()
    one_ops = set()

    del_data = removing_letter_ops(word)
    one_ops.update(del_data)

    insert_data = insert_letter_ops(word)
    one_ops.update(insert_data)

    replace_data = replace_letter_ops(word)
    one_ops.update(replace_data)

    switch_data = switch_letter_ops(word)
    one_ops.update(switch_data)

    return set(one_ops)

def edit_two_letter(word):
    two_ops = set()
    
    my_words = edit_one_letter(word)
    for i in my_words:
        data = edit_one_letter(i)
        two_ops.update(data)
    
    return set(two_ops)

def suggestions_1st_time(word,vocab_list = None):
    suggestions =set()
    processed_word = process_word(word)
    if processed_word in vocab_list:
        suggestions.update({processed_word})
    
    my_words1 = edit_one_letter(word) & vocab_list
    suggestions.update(my_words1)

    words2 = edit_two_letter(word)
    my_words2 = words2 & vocab_list
    suggestions.update(my_words2)

    words2 = list(words2)
    random.seed(42)
    random.shuffle(words2)
    words2 = words2[:10000]

    words = [edit_one_letter(x) & vocab_list for x in words2]
    suggestions.update(*words)
    req_words = list(my_words1) +list(my_words2-my_words1) +list(suggestions -my_words1.union(my_words2))

    return suggestions,req_words

def suggestions_2nd_time(word2,suggestions1, vocab_list = None,n = None):
    suggestions2 = set()
    processed_word = process_word(word2)
    if processed_word in vocab_list:
        suggestions2.update({processed_word})
    
    my_words1 = edit_one_letter(word2) & vocab_list

    suggestions2.update(my_words1)

    common_suggestions = suggestions1 & suggestions2
    len_csug = len(common_suggestions)

    if len_csug>=n:
        return list(common_suggestions)
    
    words2 = edit_two_letter(word2)
    my_words2 = words2 & vocab_list 

    suggestions2.update(my_words2)

    length = len(suggestions2.union(suggestions1))

    if length >=n:
        common_suggestions = suggestions1 & suggestions2
        suggestions = list(common_suggestions) + list(suggestions2 - common_suggestions) + list(suggestions1 - common_suggestions)
        return suggestions
    

    words2 = list(words2)
    random.seed(42)
    random.shuffle(words2)

    if len(word2)>4:
        words2 = words2[:30000]

    for i in words2:
        suggestions2.update(edit_one_letter(i) & vocab_list)
        if len(suggestions2.union(suggestions1))>=n:
            break

    
    common_suggestions = suggestions1 & suggestions2
    suggestions = list(common_suggestions) + list(suggestions2 - common_suggestions) + list(suggestions1 - common_suggestions)

    return suggestions
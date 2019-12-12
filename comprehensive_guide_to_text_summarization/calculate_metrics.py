# import...

def special_word(word):
    if word == "<q>" or word == "\n" or word == "," or word == ".":
        return True
    else:
        return False


def compute_precision(ref, candidate):
    """
    Same as Rouge-1 Precision.
    Compute the percentage of the words in the candidate summary 
    that are also present in the reference summary.

    Input:
        ref: String
        candidate: String

    Return: float
    
    Ignore non-words like <q>, \n, period, comma...

    Example:
    ref: the cat was under the bed
    candidate: the cat was found under the bed

    Result: 6/7=0.86
    """
    occurredSet = set([])
    ref_words = ref.split(' ')
    for i in range(0, len(ref_words)):
        if special_word(ref_words[i]):
            continue
        occurredSet.add(ref_words[i])
    input_words = candidate.split(' ')
    align = 0
    total = len(input_words)
    for i in range(0, len(input_words)):
        if special_word(input_words[i]):
            total = total - 1
            continue
        elif input_words[i] in occurredSet:
            align = align + 1
            continue
        else:
            continue
    return align / total
        
        

print(compute_precision("the cat was under the bed", "the cat was found under the bed"))

def compute_recall(ref, candidate):
    """
    Same as Rouge-1 Recall, or simply Rouge-1.
    Compute the percentage of the words in the reference summary 
    that are also present in the candidate summary.

    Input:
        ref: String
        candidate: String

    Return: float
    
    Ignore non-words like <q>, \n, period, comma...

    Example:
    ref: the cat was under the bed
    candidate: the cat was found under the bed

    Result: 1.0
    """
    return compute_precision(candidate, ref)

print(compute_recall("the cat was under the bed", "the cat was found under the bed"))

def compute_f1(ref, candiate):
    recall = compute_recall(ref, candidate)
    precision = compute_precision(ref, candidate)
    return 2. * recall * precision / (recall + precision)

#first ignore all special words, then consider as no special words
#ROUGE2_PRECISION([I, \n, love, <q>, python], [I, love, python]) = 1
def compute_rouge2_precision(ref, candidate):
    """
    Compute the percentage of the 2-grams words in the candidate summary 
    that are also present in the reference summary.

    Input:
        ref: String
        candidate: String

    Return: float
    
    Ignore non-words like <q>, \n, period, comma...

    Example:
    ref: the cat was under the bed
    candidate: the cat was found under the bed

    Result: 0.67
    """
    occurredSet = set([])
    ref_words_raw = ref.split(' ')
    ref_words = []
    for i in range(0, len(ref_words_raw)):
        if special_word(ref_words_raw[i]):
            continue
        else:
            ref_words.append(ref_words_raw[i])
    for i in range(0, len(ref_words) - 1):
        occurredSet.add("" + ref_words[i] + "; " + ref_words[i+1])
    input_words_raw = candidate.split(' ')
    input_words = []
    for i in range(0, len(input_words_raw)):
        if special_word(input_words_raw[i]):
            continue
        else:
            input_words.append(input_words_raw[i])
    align = 0
    total = len(input_words) - 1
    for i in range(0, len(input_words) - 1):
        if "" + input_words[i] + "; " + input_words[i+1] in occurredSet:
            align = align + 1
    return align / total
    
print(compute_rouge2_precision("the cat was under the bed", "the cat was found under the bed"))

def compute_rouge2_recall(ref, candidate):
    """
    Compute the percentage of the 2-grams words in the reference summary 
    that are also present in the candidate summary.

    Input:
        ref: String
        candidate: String

    Return: float
    
    Ignore non-words like <q>, \n, period, comma...

    Example:
    ref: the cat was under the bed
    candidate: the cat was found under the bed

    Result: 0.8
    """
    return compute_rouge2_precision(candidate, ref)

print(compute_rouge2_recall("the cat was under the bed", "the cat was found under the bed"))

def compute_rouge2_f1(ref, candiate):
    recall = compute_rouge2_recall(ref, candidate)
    precision = compute_rouge2_precision(ref, candidate)
    return 2. * recall * precision / (recall + precision)

import numpy as np

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
    ref_words = list(filter(None,ref.split(' ')))
    for i in range(0, len(ref_words)):
        if special_word(ref_words[i]):
            continue
        occurredSet.add(ref_words[i])
    input_words = list(filter(None,candidate.split(' ')))
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
    try:
        return align / total
    except:
        return 0.00
        
        


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


def compute_f1(ref, candidate):
    recall = compute_recall(ref, candidate)
    precision = compute_precision(ref, candidate)
    try:
        return 2. * recall * precision / (recall + precision)
    except:
        return 0.0 

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
    ref_words_raw = list(filter(None,ref.split(' ')))
    ref_words = []
    for i in range(0, len(ref_words_raw)):
        if special_word(ref_words_raw[i]):
            continue
        else:
            ref_words.append(ref_words_raw[i])
    for i in range(0, len(ref_words) - 1):
        occurredSet.add("" + ref_words[i] + "; " + ref_words[i+1])
    input_words_raw = list(filter(None,candidate.split(' ')))
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
    try:
        return align / total
    except:
        return 0.0
    

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


def compute_rouge2_f1(ref, candidate):
    recall = compute_rouge2_recall(ref, candidate)
    precision = compute_rouge2_precision(ref, candidate)
    try:
        return 2. * recall * precision / (recall + precision)
    except:
        return 0

def cal_metrics(metrics, original_summary, predicted_summary):
    # TODO: fill in here
    metrics["precision"].append(compute_precision(original_summary, predicted_summary))
    metrics["recall"].append(compute_recall(original_summary, predicted_summary))
    metrics["f1"].append(compute_f1(original_summary, predicted_summary))
    metrics["rouge2_precision"].append(compute_rouge2_precision(original_summary, predicted_summary))
    metrics["rouge2_recall"].append(compute_rouge2_recall(original_summary, predicted_summary))
    metrics["rouge2_f1"].append(compute_rouge2_f1(original_summary, predicted_summary))

def compute(refs, candidates):
    metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "rouge2_precision": [],
        "rouge2_recall": [],
        "rouge2_f1": [],
    }
    for i in range(len(refs)):
        cal_metrics(metrics, refs[i], candidates[i])

    for k in metrics:
        print("{}: {}".format(k, np.mean(metrics[k])))


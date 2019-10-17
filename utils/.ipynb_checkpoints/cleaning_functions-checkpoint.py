#Spelling correction
#text = TextBlob(text).correct()



#### Cleaning text
#separate words that are together. You know that there are diffrenet because there is an uppercase
def clean_text(text):
    import re
    # 1.-separate words that are together. You know that there are diffrenet because there is an uppercase
    #avoid separate abreviation
    text = re.sub(r"PC", "pc", text)
    text = re.sub(r"DVD", "dvd", text)
    text = re.sub(r"CD", "cd", text)
    text = re.sub(r"Intermet", "internet", text) 
    text = re.sub(r"<br /><br />", "", text) 
    text = re.sub(r"<br />", "", text) 
    text = re.sub(r"br", "", text) 
    #text = re.sub(r"(\w)([A-Z])", r"\1 \2", text) #separate words with uppercase UserPc = user pc
    # 2.-eliminate special characters
    text = re.sub(r"-", "", text)
    text = re.sub(r",", "", text)
    #text = re.sub(r"?", "", text)
    text = re.sub(r"\(.*\)","", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"http", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = re.sub(r"@", "at", text)
    text = re.sub('[0-9]+', "", text) # eliminate all numbers numbers
    text = text.lower()
    #3.- I'm = I am
    text = fix_abbreviation(text)
    text = re.sub('\W', ' ' , text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    # 4. spell correction
    #text = spell(text)
    
    return text

def spell_correction(text):  
    from autocorrect import spell
    #spell correction
    text = spell(text)
    return text


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"-", "")
    df[text_field] = df[text_field].str.replace(r",", "")
    df[text_field] = df[text_field].str.replace(r"?", "")
    df[text_field] = df[text_field].str.replace(r"\(.*\)","")
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"\n", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.replace('[0-9]+', "") # eliminate all numbers numbers
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r"th", "")
    df[text_field] = df[text_field].str.replace(r"h", "")
    df[text_field] = df[text_field].str.replace(r"stars", "")
    df[text_field] = df[text_field].str.replace(r"star", "")
    df[text_field] = df[text_field].str.replace(r"one", "")
    df[text_field] = df[text_field].str.replace(r"two", "")
    df[text_field] = df[text_field].str.replace(r"three", "")
    df[text_field] = df[text_field].str.replace(r"four", "")
    df[text_field] = df[text_field].str.replace(r"five", "")
    return df




'''
def standardize_text_2(text):
    text = text.str.replace(r"-", "")
    text = text.str.replace(r",", "")
    text = text.str.replace(r"?", "")
    text = text.str.replace(r"\(.*\)","")
    text = text.str.replace(r"http\S+", "")
    text = text.str.replace(r"http", "")
    text = text.str.replace(r"@\S+", "")
    text = text.str.replace(r"\n", "")
    text = text.str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    text = text.str.replace(r"@", "at")
    text = text.str.replace('[0-9]+', "") # eliminate all numbers numbers
    text = text.str.lower()
    text = text.str.replace(r"th", "")
    text = text.str.replace(r"h", "")
    text = text.str.replace(r"stars", "")
    text = text.str.replace(r"star", "")
    text = text.str.replace(r"one", "")
    text = text.str.replace(r"two", "")
    text = text.str.replace(r"three", "")
    text = text.str.replace(r"four", "")
    text = text.str.replace(r"five", "")
    return text




def clean_text_2(text):
  text = text.lower()
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", " can not ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'scuse", " excuse ", text)
  text = re.sub('\W', ' ' , text)
  text = re.sub('\s+', ' ', text)
  text = text.strip(' ')
  return text 
'''
import unicodedata
def removeAscendingChar(data):
  data=unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return data

#### lemmatization text


import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer
lemma=WordNetLemmatizer()
token=ToktokTokenizer()


def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w,'v')
        listLemma.append(x)
    return text


#-----------------------------------------------------




#fixed abbreviation
def fix_abbreviation(text):
    import re
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", " can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\bthats\b', 'that is', text)
    text = re.sub(r'\bive\b', 'i have', text)
    data_str = re.sub(r'\bim\b', 'i am', text)
    data_str = re.sub(r'\bya\b', 'yeah', data_str)
    data_str = re.sub(r'\bcant\b', 'can not', data_str)
    data_str = re.sub(r'\bdont\b', 'do not', data_str)
    data_str = re.sub(r'\bwont\b', 'will not', data_str)
    data_str = re.sub(r'\bid\b', 'i would', data_str)
    data_str = re.sub(r'wtf', 'what the fuck', data_str)
    data_str = re.sub(r'\bwth\b', 'what the hell', data_str)
    data_str = re.sub(r'\br\b', 'are', data_str)
    data_str = re.sub(r'\bu\b', 'you', data_str)
    data_str = re.sub(r'\bk\b', 'OK', data_str)
    data_str = re.sub(r'\bsux\b', 'sucks', data_str)
    data_str = re.sub(r'\bno+\b', 'no', data_str)
    data_str = re.sub(r'\bcoo+\b', 'cool', data_str)
    #data_str = re.sub(r'rt\b', '', data_str)
    data_str = data_str.strip()
    return data_str


# Part-of-Speech Tagging
def tag_and_remove(data_str):
    '''
    https://cs.nyu.edu/grishman/jet/guide/PennPOS.html
    '''
    from nltk import pos_tag
    #data_str = data_str.lower()
    cleaned_str = ' '
    # noun tags
    #nn_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    nn_tags = ['NN', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags #+ jj_tags + vb_tags

    # break string into 'words'
    text = data_str.split()

    # tag the text and keep only those with the right tags
    tagged_text = pos_tag(text)
    for tagged_word in tagged_text:
        if tagged_word[1] in nltk_tags:
            cleaned_str += tagged_word[0] + ' '

    return cleaned_str

# lemmatization
def lemmatize(data_str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    text = data_str.split()
    tagged_words = pos_tag(text)
    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lmtzr.lemmatize(word[0], pos='v')
        else:
            lemma = lmtzr.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

#polarity

def polarity_txt(text):
    from textblob import TextBlob
    return TextBlob(text).sentiment[0]
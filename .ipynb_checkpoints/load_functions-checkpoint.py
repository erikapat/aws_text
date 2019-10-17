
#first just look the title names

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

#from utils.py_functions import *
#execfile("utils/py_functions.py")
execfile("utils/cleaning_functions.py")


# Our list of functions to apply.
#create variables based in the content of the text.
transform_functions = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("CD|DVD", x)), # CD 
    lambda x: len(re.findall(r"\d+st|\d+th|\d+sd", x)), # th--> 4th, 5th or 1st or 2sd
    lambda x: len(re.findall("[A-Z]", x)), # number of uppercase letters
    lambda x: len(re.findall("[0-9]", x)), #numbers
    lambda x: len(re.findall("\d{4}", x)),
    lambda x: len(re.findall("\d$", x)), #end with number
    lambda x: len(re.findall("^\d", x)), #start with number
    lambda x: len(re.findall("[\w]+-[\w]+",x)), #words separated with -
    lambda x: len(re.findall("OLD VERSION|Old Version|old version",x)), #old version
]

transform_functions_len = [
    lambda x: len(x)
]

#---------------------------------------------------------------------------------------------------------------------------------

def merge_data(columns):
  # load data
  df_video_games      = pd.read_csv('data/amazon_reviews_us_Digital_Video_Games_v1_00.tsv', delimiter = '\t', error_bad_lines = False)
  df_software         = pd.read_csv('data/amazon_reviews_us_Software_v1_00.tsv', delimiter = '\t', error_bad_lines = False)
  df_digital_software = pd.read_csv('data/amazon_reviews_us_Digital_Software_v1_00.tsv', delimiter = '\t', error_bad_lines = False)
  
  #filter columns
  df_digital_software = df_digital_software[columns]
  df_software         = df_software[columns]
  df_video_games      = df_video_games[columns]
  
  df = pd.concat([df_digital_software, df_video_games, df_software], axis=0)
  
  #target to numeric
  dicti = {"Digital_Software": 0, "Digital_Video_Games": 1, "Software": 2}
  df['product_category_label'] = df['product_category']
  df = df.replace({"product_category": dicti})
  return df



#----------------------------------------------------------------------------------------------------------------------------------

print('Merge data')
columns = ['product_id',	'product_title',	'product_category']
df = merge_data(columns)

def cleaning_product_name():

  print('Eliminate duplicates')
  #drop duplicates
  df = df.drop_duplicates(subset = ['product_id', 'product_title', 'product_category'], keep = 'first')
  
  print('Separate pasted words')
  # apply transformations to titles
  df['product_title'] = df['product_title'].map(lambda x : sp_upper_text(x)) # separate those words with uppercase
  
  print('Create numerical variables that describe text')
  #create variables ------------------------------------------------------------------
  df_num = df[['product_id']]
  for func in transform_functions:
       df_num = pd.concat([df_num, df['product_title'].apply(func)], axis=1)
  #-------------------------------------------------------------------------------------
  #clean characters
  print('Clean special characters')
  df = standardize_text(df, "product_title")
  
  #--------------------------STOPWORDS -----------------------------------------------
  print('Eliminate stopwords')
  import nltk
  #nltk.download('stopwords')
  #nltk.download('wordnet')
  # Import stopwords with nltk.
  stop = stopwords.words('english')
  # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
  df['product_title'] = df['product_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  
  print('Eliminate english parts of words, ex. I\'m as I am')
  #more cleaning
  df['product_title'] = df['product_title'].map(lambda x : clean_text(x))
  #df['product_title'] = df['product_title'].map(lambda x : removeAscendingChar(x)) 
  print('Lemmatization')
  #lemmatization 
  #df['product_title'] = df['product_title'].map(lambda x : lemitizeWords(x))
  
  print('spelling correction')
  #spelling correction
  from textblob import TextBlob
  df['product_title'] = df['product_title'].map(lambda tweet: TextBlob(tweet).correct())
  
  return df, df_num


df, df_num = cleaning_product_name()


print(df.head())

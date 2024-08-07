#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Krithik Vasan BAskar
# #### Student ID: s3933152
# 
# Date: 01 - October - 2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * re
# * numpy
# * nlkt
# * sklearn
# * collections
# * itertools
# 
# ## Introduction
# In the realm of Natural Language Processing (NLP), understanding and representing textual data is a fundamental challenge. In this assessment task, we delve into the domain of job advertisement descriptions, seeking to transform these textual documents into meaningful and actionable feature representations. The objective is to equip ourselves with the tools necessary to extract insights from a collection of job listings.

# ## Importing libraries 

# In[1]:


# Code to import libraries
from sklearn.datasets import load_files  
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
import numpy as np
from collections import Counter
import json


# ### 1.1 Examining and loading data

# In[2]:


# Code to inspect the provided data file...
job_files = load_files(r"data")


# In[3]:


job_files['filenames']


# In[4]:


len(job_files['filenames'])


# In the data folder we have 4 sub folder called namely "Accounting_finance", "Healthcare_Nursing", "Engineering" and "Sale". In total there are 776 text files of job description.

# In[5]:


job_files['data']


# In[6]:


job_files['target']


# In[7]:


job_files['target_names']


# In[8]:


# test whether it matches, just in case
emp = 5 # an example, note we will use this example through out this exercise.
job_files['filenames'][emp], job_files['target'][emp] # from the file path we know that it's the correct class too


# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# In[9]:


jobs = []
for byte in job_files['data']:
    jobs.append(byte.decode('utf-8'))


# In[10]:


# Initialize an empty list to store dictionaries
list_of_job_dicts = []

# Initialize a variable to hold the current key
current_key = None

# Loop through each item in the job_data_list
for data in jobs:
    # Split the data into lines
    lines = data.split('\n')

    # Initialize an empty dictionary for this job posting
    job_data = {}

    # Loop through the lines and extract key-value pairs
    for line in lines:
        if ':' in line:
            # Split the line into key and value parts
            key, value = line.split(': ', 1)
            current_key = key
            job_data[key] = value
        elif current_key:
            # If there's a current key, append this line to the value
            job_data[current_key] += ' ' + line

    # Append the job_data dictionary to the list_of_job_dicts
    list_of_job_dicts.append(job_data)

# Print the list of dictionaries
list_of_job_dicts


# The provided code process a list of job data and organize it into a structured format using Python. I began by initializing an empty list called list_of_job_dicts to store dictionaries, which would represent individual job postings. Then, I iterated through each item in the jobs list, which I assumed contained textual data for various job postings.
# 
# For each job posting, I split the text into lines and created an empty dictionary called job_data to store the key-value pairs extracted from each line. I also initialized a current_key variable to keep track of the current key being processed within a job posting.
# 
# Next, I looped through the lines of the job posting, and if a line contained a colon (':'), I split the line into a key and a value based on the colon. I updated the current_key with the key, and I added the key-value pair to the job_data dictionary. If there was no colon in a line but there was a current_key, I appended the line to the value associated with that key.
# 
# Finally, after processing all lines of a job posting, I appended the job_data dictionary representing that job posting to the list_of_job_dicts. This process repeated for each job posting in the input data.
# 
# In the end, list_of_job_dicts contained a list of dictionaries, where each dictionary represented a job posting with key-value pairs extracted from the original text. This structured format made it easier to work with and analyze the job data.
# 
# 
# 
# 
# 

# In[11]:


def tokenizeReview(raw_review):
    review = raw_review.lower()
    sentences = sent_tokenize(review)
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merge them into a list of tokens
    tokenised_review = list(chain.from_iterable(token_lists))
    return tokenised_review


# tokenizeReview(raw_review): This function takes a raw text review as input and performs the following tasks:
# 
# Converts the input review to lowercase to ensure consistency in tokenization.
# Splits the review into sentences using the NLTK sent_tokenize function.
# Defines a regular expression pattern (pattern) to tokenize words. This pattern matches words containing only alphabetic characters, allowing for hyphens and apostrophes within words.
# Uses the RegexpTokenizer from NLTK to tokenize each sentence based on the defined pattern.
# Merges the token lists from all sentences into a single list called tokenised_review.
# Returns the tokenised_review, which is a list of words or tokens from the input review.

# In[12]:


def stats_print(tk_reviews):
    words = list(chain.from_iterable(tk_reviews)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of reviews:", len(tk_reviews))
    lens = [len(article) for article in tk_reviews]
    print("Average review length:", np.mean(lens))
    print("Maximun review length:", np.max(lens))
    print("Minimun review length:", np.min(lens))
    print("Standard deviation of review length:", np.std(lens))


# stats_print(tk_reviews): This function takes a list of tokenized reviews (tk_reviews) as input and calculates various statistics about the text data. It prints the following information:
# 
# Vocabulary size: The number of unique words/tokens in the entire corpus.
# Total number of tokens: The overall count of words/tokens in the corpus.
# Lexical diversity: A measure of how diverse the vocabulary is, calculated as the ratio of the vocabulary size to the total number of tokens.
# Total number of reviews: The number of reviews in the input data.
# Average review length: The average length (in tokens) of the reviews in the corpus.
# Maximum review length: The longest review in terms of tokens.
# Minimum review length: The shortest review in terms of tokens.
# Standard deviation of review length: A measure of the variability in review lengths.

# In[13]:


#Tokenizating the description.
tk_desc_job = [tokenizeReview(job_dict['Description']) for job_dict in list_of_job_dicts]  # list comprehension, generate a list of tokenized articles


# In[14]:


tk_desc_job[emp]


# In[15]:


stats_print(tk_desc_job)


# Removing all words with length less that 2

# In[16]:


tk_desc_job = [[d for d in desc if len(d) >=2] \
                      for desc in tk_desc_job]


# In[17]:


print("Tokenized review:\n",tk_desc_job[emp])


# In[18]:


stats_print(tk_desc_job)


# Removing the stopwords based on the data available inside stopwords_en.txt

# In[19]:


stopwords = []

# Open the file in read mode ('r')
with open('stopwords_en.txt', 'r') as file:
    # Read each line and append it to the 'lines' list
    for line in file:
        stopwords.append(line.strip())  # Use strip() to remove newline characters

# Print or process the list of lines
lines


# In[20]:


tk_desc_job = [[w for w in words if w not in stopwords] 
                      for words in tk_desc_job]


# In[21]:


stats_print(tk_desc_job)


# Removing the words that appears only once

# In[22]:


global_word_counts = Counter()

# Calculate term frequency (TF) for each job description and update the global word counts
for desc_tokens in tk_desc_job:
    word_counts = Counter(desc_tokens)
    global_word_counts.update(word_counts)

# Find words that appear only once (TF = 1) across all descriptions
words_to_remove = [word for word, count in global_word_counts.items() if count == 1]

words_to_remove


# In[23]:


filtered_desc_job = [[word for word in desc if word not in words_to_remove] for desc in tk_desc_job]


# In[24]:


filtered_desc_job


# In[25]:


stats_print(filtered_desc_job)


# Removing the top 50 frequent words

# In[26]:


# Calculate document frequency (DF) for each word across all job descriptions
document_frequencies = Counter()

# Count how many documents each word appears in
for desc_tokens in filtered_desc_job:
    unique_tokens = set(desc_tokens)  # Use set to count unique occurrences within each document
    document_frequencies.update(unique_tokens)

# Find the top 50 most frequent words based on DF
top_50_words = [word for word, df in document_frequencies.most_common(50)]

# Remove the top 50 most frequent words from each job description
final_desc_job = [[word for word in desc if word not in top_50_words] for desc in filtered_desc_job]
final_desc_job


# In[27]:


stats_print(final_desc_job)


# ## Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[28]:


# code to save output data...
def save_description(filename,final_desc_job):
    out_file = open(filename, 'w') # creates a txt file and open to save the reviews
    string = "\n".join([" ".join(job_desc) for job_desc in final_desc_job])
    out_file.write(string)
    out_file.close() # close the file


# In[29]:


save_description('job_description.txt',final_desc_job)


# In[30]:


# Flatten the tokenized and cleaned job descriptions into a single list
all_cleaned_tokens = [token for job_tokens in final_desc_job for token in job_tokens]

# Create a vocabulary by converting the list of unique words to a sorted list
vocabulary = sorted(set(all_cleaned_tokens))

# Define the filename for the vocabulary text file
vocabulary_filename = "vocab.txt"

# Save the vocabulary to the text file with index values starting from 0
with open(vocabulary_filename, 'w', encoding='utf-8') as vocab_file:
    for index, word in enumerate(vocabulary):
        vocab_file.write(f"{word}:{index}\n")


# In[31]:


# Open the text file for writing
with open("webindex.txt", 'w', encoding='utf-8') as webindex_file:
    for job_data in list_of_job_dicts:
        # Check if the "Webindex" key exists in the dictionary
        if "Webindex" in job_data:
            # Write the value of "Webindex" to the text file, followed by a newline
            webindex_file.write(job_data["Webindex"] + '\n')


# In[32]:


target_data = np.array(job_files['target'], dtype=int)
with open('target.txt', 'w') as file:
    for item in target_data:
        file.write(str(item) + '\n')


# In[33]:


# Open the text file for writing
with open("title.txt", 'w', encoding='utf-8') as title_file:
    for job_data in list_of_job_dicts:
        # Check if the "Title" key exists in the dictionary
        if "Title" in job_data:
            # Write the value of "Title" to the text file, followed by a newline
            title_file.write(job_data["Title"] + '\n')


# In[34]:


title_desc = [(job['Title'] + " " + job['Description']) for job in list_of_job_dicts]


# In[35]:


title_desc


# In[36]:


with open('title_desc.txt', 'w', encoding='utf-8') as file:
    for item in title_desc:
        file.write(item + '\n')


# ## Summary
# 

# In Task 1, the goal was to perform basic text pre-processing on a given dataset, specifically focusing on the description field of job advertisements. The following steps were executed:
# 
# 1. **Information Extraction**: Information was extracted from each job advertisement, and the pre-processing steps were applied to their descriptions.
# 
# 2. **Tokenization**: The descriptions were tokenized using the provided regular expression, r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", which allowed for words like "it's" or "well-known" to be treated as single tokens.
# 
# 3. **Lowercasing**: All words were converted to lowercase to ensure uniformity.
# 
# 4. **Removing Short Words**: Words with a length of less than 2 characters were removed as they typically do not carry meaningful information.
# 
# 5. **Stopword Removal**: A list of stopwords from stopwords_en.txt was used to remove common words that don't contribute significantly to the meaning of the text.
# 
# 6. **Removing Low-Frequency Words**: Words that appeared only once in the entire document collection were removed based on term frequency. This helped reduce noise in the data.
# 
# 7. **Removing Top Frequent Words**: The top 50 most frequent words based on document frequency were removed to further clean the data.
# 
# 8. **Saving Preprocessed Data**: The preprocessed job advertisement text and information were saved in a specified format in text file(s).
# 
# 9. **Building Vocabulary**: A vocabulary of cleaned job advertisement descriptions was built and saved in vocab.txt. The vocabulary contained words sorted in alphabetical order, with index values starting from 0. This vocabulary is essential for interpreting the sparse encoding of the data.
# 
# The output of Task 1 included the following files:
# 
# - **vocab.txt**: This file contained the unigram vocabulary, with each word on a separate line in the format "word_string:word_integer_index," sorted alphabetically.
# 
# Task 1 was completed successfully, ensuring that the text data from job advertisements was preprocessed and organized for further analysis or use in subsequent tasks.

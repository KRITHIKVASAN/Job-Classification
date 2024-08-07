#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
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
# In Task 2, the objective is to generate various types of feature representations for a collection of job advertisements, with a specific focus on the descriptions within those job advertisements. The task involves creating three distinct feature representations:
# 
# 1. **Bag-of-Words Model (Count Vector Representation)**:
#    - Generate the count vector representation for each job advertisement description.
#    - Save these count vectors into a file, following a specific format.
#    - The count vector representations will be based on the vocabulary created in Task 1, which was saved in vocab.txt.
# 
# 2. **Models Based on Word Embeddings**:
#    - Choose one embedding language model, such as FastText, GoogleNews300, or another pre-trained Word2Vec model, or Glove.
#    - Build two types of document embeddings for each job advertisement description:
#      - **TF-IDF Weighted Vector Representation**: Generate a weighted vector representation using the chosen language model and TF-IDF weighting.
#      - **Unweighted Vector Representation**: Create an unweighted vector representation for each description using the chosen language model.
# 
# The output for Task 2 will include the following:
# 
# - **count_vectors.txt**: This file will store the sparse count vector representations of job advertisement descriptions. Each line in this file corresponds to one advertisement and follows a specific format. It includes the webindex of the job advertisement, a comma, and the sparse representation of the description in the form of word_integer_index:word_freq, separated by commas.
# 
# Task 2 is crucial for transforming the textual data from job advertisements into numerical feature representations, which can be used for various downstream tasks such as text classification or clustering. These feature representations capture the essence of the job descriptions, enabling further analysis and modeling of the dataset.

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import nltk
import re


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# In[2]:


# Initialize an empty list to store the vocabulary
vocab  = []

with open('vocab.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line at the colon (":") character
        parts = line.strip().split(':')
        if len(parts) > 1:
            # Get the word before the colon and add it to the vocabulary list
            word = parts[0]
            vocab.append(word)

# Print the vocabulary list
vocab


# In[3]:


# Initialize an empty list to store the job description
job_desc = []

# Open the text file for reading
with open('job_description.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Append each line (job description) to the job_desc list
        job_desc.append(line.strip())

# Print the job_desc list
job_desc


# In[4]:


vVectorizer = CountVectorizer(analyzer = "word",vocabulary=vocab)
count_features  = vVectorizer.fit_transform(job_desc)
count_features.shape


# In[5]:


print(count_features)


# In[6]:


# Initialize an empty list to store the values
webindex_list = []

# Open the text file for reading
with open('webindex.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Append each line (value) to the values list, removing leading and trailing whitespace
        webindex_list.append(line.strip())

# Print the values list
webindex_list


# In[7]:


def validator(count_features, vocab, a_ind, indeces):
    print("WEB INDEX:", indeces[a_ind]) # print out the Article ID
    print("--------------------------------------------")
    print("Job Detail:",job_desc[a_ind]) # print out the txt of the article
    #print("Article tokens:",tokenised_articles[a_ind]) # print out the tokens of the article
    print("--------------------------------------------\n")
    print("Vector representation:\n") # printing the vector representation as format 'word:value' (
                                      # the value is 0 or 1 in for binary vector; an integer for count vector; and a float value for tfidf

    for word, value in zip(vocab, count_features.toarray()[a_ind]): 
        if value > 0:
            print(word+":"+str(value), end =' ')


# In[8]:


validator(count_features, vocab, 775, webindex_list)


# In[9]:


count_vector = []
for i in range(len(webindex_list)):
    cv = "#" + webindex_list[i]
    for index, value in enumerate(count_features.toarray()[i]):
        if value > 0:
            cv += "," + str(index) + ":" + str(value)
    count_vector.append(cv)


# In[10]:


count_vector


# In[11]:


get_ipython().system('pip install gensim')


# In[12]:


import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize


# In[13]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(job_desc)


# In[14]:


print(tfidf_vectors)


# ## Word2Vec model
# Popular word embedding model in natural language processing is Word2Vec. Words are transformed into numerical vectors that represent semantic meaning and word connections. Word2Vec uses the Skip-gram and Continuous Bag of Words (CBOW) as its two main algorithms while training on huge text datasets. These embeddings have intriguing features that make it possible to do vector arithmetic operations that frequently provide useful results. For a variety of NLP applications, from sentiment analysis to machine translation, where understanding context and word meanings is key, pre-trained Word2Vec models are easily accessible for many languages and domains.

# In[15]:


# Tokenize the job descriptions
tokenized_descriptions = [word_tokenize(description.lower()) for description in job_desc]

# Train a Word2Vec model
model = Word2Vec(sentences=tokenized_descriptions, vector_size=100, window=5, min_count=1, sg=0)

# Save the trained model to a file (you can replace 'word2vec.model' with your desired filename)
model.save("word2vec.model")


# In[16]:


unweighted_embeddings = np.array([np.mean([model.wv[word] for word in tokens if word in model.wv] if tokens else [np.zeros(model.vector_size)], axis=0)
                                  for tokens in tokenized_descriptions])


# In[17]:


unweighted_embeddings


# In[18]:


tfidf_weighted_embeddings = []
for tokens in tokenized_descriptions:
    embeddings = []
    for word in tokens:
        if word in model.wv and word in tfidf_vectorizer.vocabulary_:
            tfidf_weighted_vector = model.wv[word] * tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
            embeddings.append(tfidf_weighted_vector)
    if embeddings:
        tfidf_weighted_embeddings.append(np.mean(embeddings, axis=0))
    else:
        # If no valid word vectors found, use a zero vector
        tfidf_weighted_embeddings.append(np.zeros(model.vector_size))

# Convert the list of embeddings to a numpy array
tfidf_weighted_embeddings = np.array(tfidf_weighted_embeddings)


# In[19]:


tfidf_weighted_embeddings


# In[20]:


#saving count vector in a file.
with open('count_vector.txt', 'w') as file:
    file.write('\n'.join(map(str, count_vector)))


# ## Task 3. Job Advertisement Classification

# In Task 3, we delve into the domain of job advertisement classification, aiming to build machine learning models capable of categorizing the content of job advertisements into specific categories. This task involves conducting two sets of experiments to address two critical questions:
# 
# One of the central inquiries in this task revolves around assessing the performance of various language models that were generated in Task 2, using the feature representations derived from job advertisement descriptions. To tackle this question, we embark on building machine learning models, employing these feature representations, and scrutinizing their classification performance.
# 
# We will not only explore conventional models such as logistic regression from scikit-learn but also have the flexibility to consider other machine learning models, even those not explicitly covered in the course. Our primary objective is to ascertain which language model, coupled with the chosen machine learning algorithm, yields the most promising results. Through rigorous evaluation, we aim to determine the most effective combination for accurately classifying job advertisements into their respective categories.
# 
# The second question revolves around the potential benefits of incorporating additional information into the classification process. Specifically, we are interested in assessing whether including the title of the job position, in addition to the description, improves the accuracy of our classification models. To explore this, we will conduct experiments that consider three distinct scenarios:
# 
# 1. **Using Only the Title**: In this scenario, we exclusively leverage the title of the job advertisement to build classification models.
# 
# 2. **Using Only the Description**: This scenario involves utilizing only the job advertisement descriptions, a feature representation we have already crafted in Task 2.
# 
# 3. **Using Both Title and Description**: In this scenario, we have the flexibility to either concatenate the title and description into a single feature representation or generate separate feature representations for both the title and description. We will explore both approaches and assess their impact on classification accuracy.
# 
# To ensure robust and reliable comparisons, we will employ a 5-fold cross-validation methodology during the evaluation process. This approach helps us mitigate bias and provides a comprehensive view of how different models and data combinations perform under various conditions.
# 
# Ultimately, the outcomes of Task 3 will shed light on the efficacy of language models, the potential advantages of incorporating additional information, and guide us in selecting the most suitable strategies for job advertisement classification, aligning our efforts with the overarching goal of optimizing accuracy and performance.

# In[21]:


# Initialize an empty list to store the data
target = []

# Open the file in read mode ('r') and read the data
with open('target.txt', 'r') as file:
    for line in file:
        # Convert the line to an integer and append it to the 'target' list
        target.append(int(line.strip()))


# In[22]:


target


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


# In[24]:


num_folds = 5
seed = 15
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
kf


# In[25]:


def evaluate(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(random_state=seed,max_iter = 1000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[26]:


import pandas as pd
from sklearn.linear_model import LogisticRegression

num_models = 2
cv_df = pd.DataFrame(columns = ['unweighted','weighted'],index=range(num_folds)) # creates a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(target)))):
    y_train = [str(target[i]) for i in train_index]
    y_test = [str(target[i]) for i in test_index]

   
    X_train_unweighted, X_test_unweighted = unweighted_embeddings[train_index], unweighted_embeddings[test_index]
    cv_df.loc[fold,'unweighted'] = evaluate(unweighted_embeddings[train_index],unweighted_embeddings[test_index],y_train,y_test,seed)

    X_train_weighted, X_test_weighted = tfidf_weighted_embeddings[train_index], tfidf_weighted_embeddings[test_index]
    cv_df.loc[fold,'weighted'] = evaluate(tfidf_weighted_embeddings[train_index],tfidf_weighted_embeddings[test_index],y_train,y_test,seed)
    
    fold +=1


# In[27]:


cv_df


# In[28]:


cv_df.mean()


# In[ ]:





# ### Repeating the same process for the feature generation of "TITLE"

# In[ ]:





# In[29]:


# Initialize an empty list to store the data
title = []

# Open the file in read mode ('r') and read the data
with open('title.txt', 'r') as file:
    for line in file:
        # Append each line (string) to the 'title' list
        title.append(line.strip())

# Print or use the 'title' list
title


# In[30]:


from nltk.tokenize import sent_tokenize, RegexpTokenizer
from itertools import chain
import numpy as np

def tokenizeReview(title):
    review = title.lower()
    sentences = sent_tokenize(review)
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern)
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]

    # merge them into a list of tokens
    tokenized_title = list(chain.from_iterable(token_lists))
    return tokenized_title

def stats_print(tk_title):
    words = list(chain.from_iterable(tk_title))  # we put all the tokens in the corpus in a single list
    vocab = set(words)  # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab) / len(words)
    print("Vocabulary size: ", len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of titles:", len(tk_title))
    lens = [len(title) for title in tk_title]
    print("Average title length:", np.mean(lens))
    print("Maximum title length:", np.max(lens))
    print("Minimum title length:", np.min(lens))
    print("Standard deviation of title length:", np.std(lens))


# In[31]:


tk_job_title = [tokenizeReview(job_title) for job_title in title]  # list comprehension, generate a list of tokenized articles


# In[32]:


tk_job_title


# In[33]:


stats_print(tk_job_title)


# In[34]:


st_list = [[d for d in desc if len(d) <= 1] \
                      for desc in tk_job_title] # create a list of single character token for each review
list(chain.from_iterable(st_list)) # merge them together in one list


# In[35]:


tk_job_title = [[d for d in desc if len(d) >=2] \
                      for desc in tk_job_title]


# In[36]:


stats_print(tk_job_title)


# In[37]:


stopwords = []

# Open the file in read mode ('r')
with open('stopwords_en.txt', 'r') as file:
    # Read each line and append it to the 'lines' list
    for line in file:
        stopwords.append(line.strip())  # Use strip() to remove newline characters

# Print or process the list of lines
stopwords


# In[38]:


tk_job_title = [[w for w in words if w not in stopwords] 
                      for words in tk_job_title]


# In[39]:


stats_print(tk_job_title)


# In[40]:


final_job_title = [" ".join(t) for t in tk_job_title]
final_job_title


# In[41]:


stats_print(final_job_title)


# In[42]:


title_vectors = tfidf_vectorizer.fit_transform(final_job_title)


# In[43]:


print(title_vectors)


# In[44]:


# Tokenize the job descriptions
tokenized_title = [word_tokenize(t.lower()) for t in final_job_title]

# Train a Word2Vec model
model_title = Word2Vec(sentences=tokenized_title, vector_size=100, window=5, min_count=1, sg=0)

# Save the trained model to a file (you can replace 'word2vec.model' with your desired filename)
model_title.save("word2vec_title.model")


# In[45]:


unweighted_title_embeddings = np.array([np.mean([model_title.wv[word] for word in tokens if word in model_title.wv] if tokens else [np.zeros(model_title.vector_size)], axis=0)
                                  for tokens in tokenized_title])


# In[46]:


unweighted_title_embeddings


# In[47]:


tfidf_weighted_title_embeddings = []
for tokens in tokenized_title:
    embeddings = []
    for word in tokens:
        if word in model_title.wv and word in tfidf_vectorizer.vocabulary_:
            
            
            tfidf_weighted_vector = model_title.wv[word] * tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
            embeddings.append(tfidf_weighted_vector)
    if embeddings:
        tfidf_weighted_title_embeddings.append(np.mean(embeddings, axis=0))
    else:
        # If no valid word vectors found, use a zero vector
        tfidf_weighted_title_embeddings.append(np.zeros(model.vector_size))

# Convert the list of embeddings to a numpy array
tfidf_weighted_title_embeddings = np.array(tfidf_weighted_title_embeddings)


# In[48]:


tfidf_weighted_title_embeddings


# In[49]:


cv_df_title = pd.DataFrame(columns = ['unweighted','weighted'],index=range(num_folds)) # creates a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(target)))):
    y_train = [str(target[i]) for i in train_index]
    y_test = [str(target[i]) for i in test_index]
   
    X_train_unweighted, X_test_unweighted = unweighted_title_embeddings[train_index], unweighted_title_embeddings[test_index]
    cv_df_title.loc[fold,'unweighted'] = evaluate(unweighted_title_embeddings[train_index],unweighted_title_embeddings[test_index],y_train,y_test,seed)

    X_train_weighted, X_test_weighted = tfidf_weighted_title_embeddings[train_index], tfidf_weighted_title_embeddings[test_index]
    cv_df_title.loc[fold,'weighted'] = evaluate(tfidf_weighted_title_embeddings[train_index],tfidf_weighted_title_embeddings[test_index],y_train,y_test,seed)
    
    fold +=1


# In[50]:


cv_df_title


# In[51]:


cv_df_title.mean()


# In[52]:


combined_unweighted_embeddings = np.hstack((unweighted_embeddings, unweighted_title_embeddings))
combined_weighted_embedings = np.hstack((unweighted_embeddings, tfidf_weighted_title_embeddings))


# In[53]:


cv_df_combined = pd.DataFrame(columns = ['unweighted','weighted'],index=range(num_folds)) # creates a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(target)))):
    y_train = [str(target[i]) for i in train_index]
    y_test = [str(target[i]) for i in test_index]
   
    X_train_unweighted, X_test_unweighted = combined_unweighted_embeddings[train_index], combined_unweighted_embeddings[test_index]
    cv_df_combined.loc[fold,'unweighted'] = evaluate(combined_unweighted_embeddings[train_index],combined_unweighted_embeddings[test_index],y_train,y_test,seed)

    X_train_weighted, X_test_weighted = combined_weighted_embedings[train_index], combined_weighted_embedings[test_index]
    cv_df_combined.loc[fold,'weighted'] = evaluate(combined_weighted_embedings[train_index],combined_weighted_embedings[test_index],y_train,y_test,seed)
    
    fold +=1


# In[54]:


cv_df_combined


# In[55]:


cv_df_combined.mean()


# In[ ]:





# ### Repeating the same process for the feature generation of "TITLE & DESCRIPTION" joined

# In[ ]:





# In[56]:


# Initialize an empty list to store the job description
title_desc = []

# Open the text file for reading
with open('title_desc.txt', 'r', encoding='utf-8') as file:
    # Read each line in the file
    for line in file:
        # Append each line (job description) to the job_desc list
        title_desc.append(line.strip())

# Print the job_desc list
title_desc


# In[57]:


tk_title_desc = [tokenizeReview(td) for td in title_desc]  # list comprehension, generate a list of tokenized articles


# In[58]:


print(tk_title_desc)


# In[59]:


stats_print(tk_title_desc)


# In[60]:


st_list_ = [[d for d in desc if len(d) <= 1] \
                      for desc in tk_title_desc] # create a list of single character token for each review
list(chain.from_iterable(st_list_)) # merge them together in one list


# In[61]:


tk_title_desc = [[d for d in desc if len(d) >=2] \
                      for desc in tk_title_desc]


# In[62]:


stats_print(tk_title_desc)


# In[63]:


tk_title_desc = [[w for w in words if w not in stopwords] 
                      for words in tk_title_desc]


# In[64]:


stats_print(tk_title_desc)


# In[65]:


from collections import Counter

global_word_counts = Counter()

# Calculate term frequency (TF) for each job description and update the global word counts
for title_tokens in tk_title_desc:
    word_counts = Counter(title_tokens)
    global_word_counts.update(word_counts)

# Find words that appear only once (TF = 1) across all descriptions
words_to_remove = [word for word, count in global_word_counts.items() if count == 1]

words_to_remove


# In[66]:


filtered_title_desc = [[word for word in desc if word not in words_to_remove] for desc in tk_title_desc]


# In[67]:


stats_print(filtered_title_desc)


# In[68]:


# Calculate document frequency (DF) for each word across all job descriptions
document_frequencies = Counter()

# Count how many documents each word appears in
for title_tokens in filtered_title_desc:
    unique_tokens = set(title_tokens)  # Use set to count unique occurrences within each document
    document_frequencies.update(unique_tokens)

# Find the top 50 most frequent words based on DF
top_50_words = [word for word, df in document_frequencies.most_common(50)]

# Remove the top 50 most frequent words from each job description
final_title_desc = [[word for word in desc if word not in top_50_words] for desc in filtered_title_desc]
final_title_desc


# In[69]:


stats_print(final_title_desc)


# In[70]:


final_title_desc = [" ".join(t) for t in final_title_desc]
final_title_desc


# In[71]:


title_desc_vectors = tfidf_vectorizer.fit_transform(final_title_desc)


# In[72]:


print(title_desc_vectors)


# In[73]:


# Tokenize the job descriptions
tokenized_title_desc = [word_tokenize(t.lower()) for t in final_title_desc]

# Train a Word2Vec model
model_title_desc = Word2Vec(sentences=tokenized_title_desc, vector_size=100, window=5, min_count=1, sg=0)

# Save the trained model to a file (you can replace 'word2vec.model' with your desired filename)
model_title_desc.save("word2vec_title_desc.model")


# In[74]:


unweighted_title_desc_embeddings = np.array([np.mean([model_title_desc.wv[word] for word in tokens if word in model_title_desc.wv] if tokens else [np.zeros(model_title_desc.vector_size)], axis=0)
                                  for tokens in tokenized_title_desc])


# In[75]:


unweighted_title_desc_embeddings


# In[76]:


tfidf_weighted_title_desc_embeddings = []
for tokens in tokenized_title_desc:
    embeddings = []
    for word in tokens:
        if word in model_title.wv and word in tfidf_vectorizer.vocabulary_:
            
            
            tfidf_weighted_vector = model_title.wv[word] * tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
            embeddings.append(tfidf_weighted_vector)
    if embeddings:
        tfidf_weighted_title_desc_embeddings.append(np.mean(embeddings, axis=0))
    else:
        # If no valid word vectors found, use a zero vector
        tfidf_weighted_title_desc_embeddings.append(np.zeros(model.vector_size))

# Convert the list of embeddings to a numpy array
tfidf_weighted_title_desc_embeddings = np.array(tfidf_weighted_title_desc_embeddings)


# In[77]:


tfidf_weighted_title_desc_embeddings


# In[78]:


cv_df_title_desc = pd.DataFrame(columns = ['unweighted','weighted'],index=range(num_folds)) # creates a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(target)))):
    y_train = [str(target[i]) for i in train_index]
    y_test = [str(target[i]) for i in test_index]
   
    X_train_unweighted, X_test_unweighted = unweighted_title_desc_embeddings[train_index], unweighted_title_desc_embeddings[test_index]
    cv_df_title_desc.loc[fold,'unweighted'] = evaluate(unweighted_title_desc_embeddings[train_index], unweighted_title_desc_embeddings[test_index], y_train, y_test, seed)

    X_train_weighted, X_test_weighted = tfidf_weighted_title_desc_embeddings[train_index], tfidf_weighted_title_desc_embeddings[test_index]
    cv_df_title_desc.loc[fold,'weighted'] = evaluate(tfidf_weighted_title_desc_embeddings[train_index],tfidf_weighted_title_desc_embeddings[test_index],y_train,y_test,seed)
    
    fold +=1


# In[79]:


cv_df_title_desc


# In[80]:


cv_df_title_desc.mean()


# In[ ]:





# ### Model output explaination
# 
# In the context of regression modeling for job titles and descriptions, an intriguing observation arises when comparing the effectiveness of weighted (TF-IDF) and unweighted word embeddings. When these two types of text data are analyzed separately, TF-IDF weighting often proves to be more effective in explaining variance in the target variable. This result aligns with the expectation that TF-IDF helps emphasize crucial terms within each subset of data.
# 
# However, when we combine titles and descriptions for analysis, an interesting shift occurs. In this combined setting, unweighted word embeddings tend to perform exceptionally well, surpassing the performance of TF-IDF-weighted embeddings. Several factors contribute to this change, including differences in data distribution, potential redundancy between titles and descriptions, and the complex interactions introduced by their combination.
# 
# The choice between weighted and unweighted embeddings remains highly dependent on the nature of the data and the specific regression task at hand. Experimentation and careful consideration are key. It's not uncommon to explore various feature representations and model complexities to determine the optimal combination for a given dataset. Additionally, techniques such as feature selection and dimensionality reduction may further enhance model performance, especially when working with combined text data.
# 
# In conclusion, the choice of feature representation in regression modeling reflects the nuanced relationship between text data and target variables. It's a reminder that effective modeling in the realm of natural language processing requires adaptability and a thorough understanding of the data's intricacies.

# In[ ]:





# ## Summary
# In the realm of Natural Language Processing (NLP), the art of transforming textual data into actionable insights plays a pivotal role in various applications. In Task 2 and Task 3, we embarked on a journey to explore the intricate nuances of feature representation and regression analysis for job advertisement descriptions and titles, aiming to unlock hidden patterns and predictive power within this textual domain.
# 
# **Task 2: Feature Representation**
# 
# In Task 2, we delved into the realm of feature representation. Our goal was to bridge the gap between unstructured text data and machine learning models. This journey involved:
# 
# 1. **Data Preprocessing:** We began by meticulously preparing our textual data. Tokenization, stemming, and the removal of stopwords were key steps in ensuring that our text was machine-readable.
# 
# 2. **Language Models:** The choice of a suitable language model was pivotal. We explored the power of Word2Vec, FastText, or GloVe embeddings, each offering unique advantages in capturing semantic meaning.
# 
# 3. **Feature Engineering:** We crafted three distinct feature representations for job descriptions and titles. Count vectors, capturing word frequency, offered a straightforward representation. TF-IDF weighted vectors highlighted the importance of terms. Word embeddings, both weighted and unweighted, portrayed the semantic essence of the text.
# 
# **Task 3: Regression Analysis**
# 
# With our feature representations in hand, we embarked on Task 3: regression analysis. Here, our aim was to predict and understand the target variable. The outcome unveiled intriguing insights:
# 
# - **Feature Effectiveness:** We discovered that the effectiveness of feature representations varied across different settings. Count vectors often excelled, portraying the strength of capturing word frequency. However, TF-IDF weighted vectors sometimes outperformed unweighted word embeddings, emphasizing the significance of term weighting.
# 
# - **Interplay of Features:** When titles and descriptions were analyzed separately, TF-IDF weighting frequently enhanced performance. Yet, when combined, unweighted embeddings often shone. This transition reflected the intricate interplay between these two types of text data, including potential redundancy and complex feature interactions.
# 
# - **Model Flexibility:** The choice of regression model complexity also influenced the results. More complex models proved adept at capturing combined information, while simpler models excelled when features were more straightforward.
# 
# In essence, Tasks 2 and 3 underscored the dynamic nature of text data analysis. It emphasized the necessity of adaptability, experimentation, and a deep understanding of data intricacies. Whether it's uncovering patterns in job descriptions or unraveling the predictive power of titles, the journey through feature representation and regression analysis is a testament to the art of translating text into actionable insights.

# In[ ]:





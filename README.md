
## Topic Modelling - LDA model with Pyspark

Attempts to use Topic modelling method to automatically extract key topics from descriptions. 


### Exploration
After downloading parquet file, performed simple data exploration. 

*Noticed:*
1. NO NULL value
2. "Genre" columns is a list between range of 1~3
3. "Description" column is free text, consists of many languages, including English. 

### Starting point
Will Start with English description first. Then expand the model to other languages in later phase. 

### Implementation Planning
1. Detect language used in the "description", and create new column "lang_code" storing the code. For the ease of later analysis. 
2. Text cleaning, removing punctuation/stopwords, then get it tokenized.
3. Form tf-idf countvector on whole dataset.
4. Base on the vector, build LDA model on each row of record. Extract five (changable) topic keywords in each of the top five (changable) topics. 

------ Until Now ---------
(Next step)
5. Combine the result back into the dataset, with each column has a list of keywords.

#### Thoughts:
1. Besides topic modelling, to realize keyword auto tagging, can also use "genre" column, to find out the word(s) with the closest similarity from these genre tags. 
2. 


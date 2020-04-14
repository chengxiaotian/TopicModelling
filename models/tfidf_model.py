#!/usr/bin/env python
# coding: utf-8
"""
TFIDF model for different languages

Current support languages:
- English
- Chinese
- Bahasa

"""

from langdetect import detect
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer
import nltk
from nltk.stem import WordNetLemmatizer 
from pyspark.sql.types import *
import pyspark.sql.functions as F
import array as arr 
from pyspark.ml.feature import StopWordsRemover
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from pyspark.sql.functions import regexp_replace
import jieba
import pyspark.sql.functions as F
from pyspark.sql.types import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pyspark.ml.feature import Tokenizer

"""
Common function for building TFIDF model
input: 
- corpus list for train the count vector and tfidf transformer
- pandas dataframe to return the final keyword list result
- number of topn prefered

"""
def get_tfidf(corpus_list,df_pd_desc_final,topn):
    # Step 4. TF-IDF countvector, Transform documents to document-term matrix

    cv = CountVectorizer(max_df=0.95) #ignore terms that have a document frequency strictly higher than the given threshold
    dtm = cv.fit_transform(corpus_list)
    feature_names = cv.get_feature_names()

    # Step 5. TfidfTransformer to compute the IDF
    tfidfTransformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidfTransformer.fit(dtm)
    
    def get_keyword_list(doc):
        # generate tfidf vector for given doc
        tf_idf_vector = tfidfTransformer.transform(cv.transform([doc]))

        # Return a C00rdinate representation of this matrix
        coo_matrix = tf_idf_vector.tocoo()
        # sorted_items = sort_coo(coo_matrix)

        # get Top-n index list
        index_sort = np.argsort(coo_matrix.data)[-topn:]

        keyword_list = []
        # extract keywords
        for idx in index_sort[::-1]: 
            keyword_list.append(feature_names[coo_matrix.col[idx]])

        return keyword_list

    # Step 6. Get consolidated keyword list for entire dataset
    

    df_pd_desc_final['description_keyword_list']=df_pd_desc_final['description_final'].apply(get_keyword_list)
    return df_pd_desc_final

"""
English TFIDF model
input: 
- spark data, for the portion of specific language
- number of topn prefered

"""
class tfidf_eng_model:
    
    def __init__(self,spark_data,topn):
        self.df_spark = spark_data
        self.topn = topn
    
    def get_pd_keyword(self):
        
        df_spark = self.df_spark 

        # Step 1. Text cleasing with punctuations
        
        REGEX = '[_,?\\-.!?@#$%^&*+\/\d]'
        df_spark = df_spark.withColumn("description_clean",regexp_replace(df_spark.description,REGEX,' '))


        # Step 2. Tokenization
        # df_spark = df_spark.drop("description_token")
        
        tokenizer = Tokenizer(inputCol='description_clean',outputCol='description_token')
        df_spark = tokenizer.transform(df_spark)


        # Stemming
        # nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer() 
        
        def lemm_function(list):
            list_clean = []
            for item in list:
                list_clean.append(lemmatizer.lemmatize(item))

            return list_clean

        udf_lemm_function= F.udf(lemm_function, ArrayType(StringType()))

        df_spark = df_spark.withColumn("description_lemm",udf_lemm_function(df_spark.description_token))

        # Step 3. Remove stopword

        stopwords_list = StopWordsRemover.loadDefaultStopWords("english")
        stopwords_customize_list = ["app","apps"]
        stopwords_list = np.append(stopwords_list,stopwords_customize_list)

        stopwords = StopWordsRemover(inputCol="description_lemm",outputCol="description_no_stop",stopWords=stopwords_list)
        stopwords.getStopWords()
        df_spark = stopwords.transform(df_spark)


        df_pd_desc_final = df_spark.toPandas()

        # ### Note: IDF vector must be trained with large corpus, otherwise lose the advance of IDF


        # get the "description" column
        joinF= lambda x:" ".join(x)
        df_pd_desc_final["description_final"] = df_pd_desc_final["description_no_stop"].apply(joinF)

        corpus_list = df_pd_desc_final["description_final"].tolist()

        df_pd_desc_final = get_tfidf(corpus_list,df_pd_desc_final,self.topn)
        
        return df_pd_desc_final
    
"""
Chinese TFIDF model
input: 
- spark data, for the portion of specific language
- number of topn prefered

"""     
class tfidf_cn_model:
    def __init__(self,spark_data,topn):
        self.df_spark = spark_data
        self.topn = topn
        
    def get_pd_keyword(self):
        
        df_spark_cn = self.df_spark
        # ### #1: customize punctuation to chinese
        # Step 1. Text cleasing with punctuations
        # df_spark_cn = df_spark_cn.drop("description_clean")
        REGEX = '[「」【】；！_，、~：。？\-\\.!?@#$%^&*+/\d]' 
        df_spark_cn = df_spark_cn.withColumn("description_clean",regexp_replace(df_spark_cn.description,REGEX,' '))

        # ### #2: get tokens in Chinese
        # Step 2. Tokenization

        def tokenF(str):
            return ",".join(jieba.cut(str))

        udf_tokenF = F.udf(tokenF,StringType())

        df_spark_cn = df_spark_cn.withColumn("description_final",udf_tokenF(df_spark_cn.description_clean))


        # ### #3: prepare stopwords in Chinese
        # Step 3. prepare stopword list
        stopword_cn = pd.read_excel("stopword_cn.xlsx")

        # prepare for stopwword list
        splitF = lambda x: x.split(",")
        stopword_cn['Stopwords_list'] = stopword_cn['Stopwords'].apply(splitF)

        stopword_list = []
        for list in stopword_cn['Stopwords_list']:
            stopword_list = stopword_list+list


        # ### Build TF-IDF model
        pd_cn = df_spark_cn.toPandas()
        corpus_list = pd_cn['description_final'].tolist()

        pd_cn = get_tfidf(corpus_list,pd_cn,self.topn)
        
        return pd_cn


"""
Bahasa TFIDF model
input: 
- spark data, for the portion of specific language
- number of topn prefered

"""     
class tfidf_id_model:
    def __init__(self,spark_data,topn):
        self.df_spark = spark_data
        self.topn = topn
        
    def get_pd_keyword(self):
        
        df_spark = self.df_spark
        # Step 1. Text cleasing with punctuations
        REGEX = '[()`_,.?!?@#$%^&*+\/\d]'
        df_spark = df_spark.withColumn("description_clean",regexp_replace(df_spark.description,REGEX,' '))


        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stemF = lambda x: stemmer.stem(x)


        udf_stemF = F.udf(stemF,StringType())
        df_spark = df_spark.withColumn("description_stem",udf_stemF(df_spark.description_clean))


        # Step 2. Tokenization
        df_spark = df_spark.drop("description_token")
        tokenizer = Tokenizer(inputCol='description_stem',outputCol='description_token')
        df_spark = tokenizer.transform(df_spark)


        # Step 3. prepare stopword list
        stopword_id = pd.read_csv("stopword_id.csv",header = None,names=(['stopword']))


        stopword_list = [x for x in stopword_id['stopword'].values]

        df_pd_desc_final = df_spark.toPandas()

        # ### Note: IDF vector must be trained with large corpus, otherwise lose the advance of IDF

        # get the "description" column
        joinF= lambda x:" ".join(x)
        df_pd_desc_final["description_final"] = df_pd_desc_final["description_token"].apply(joinF)

        corpus_list = df_pd_desc_final['description_final'].tolist()

        pd_id = get_tfidf(corpus_list,df_pd_desc_final,self.topn)
        
        return pd_id

        
        
class tfidf_rest_model:
    
    def __init__(self,spark_data,spark,topn):
        self.df_spark = spark_data
        self.spark = spark
        self.topn = topn
    
    def get_pd_keyword(self):
        
        df_spark = self.df_spark 

        # Step 1. Text cleasing with punctuations
        
        REGEX = '[_,?\\-.!?@#$%^&*+\/\d]'
        df_spark = df_spark.withColumn("description_clean",regexp_replace(df_spark.description,REGEX,' '))


        # Step 2. Tokenization
        # df_spark = df_spark.drop("description_token")
        
        tokenizer = Tokenizer(inputCol='description_clean',outputCol='description_token')
        df_spark = tokenizer.transform(df_spark)


        # Stemming
        # nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer() 
        
        def lemm_function(list):
            list_clean = []
            for item in list:
                list_clean.append(lemmatizer.lemmatize(item))

            return list_clean

        udf_lemm_function= F.udf(lemm_function, ArrayType(StringType()))

        df_spark = df_spark.withColumn("description_lemm",udf_lemm_function(df_spark.description_token))

        # Step 3. Remove stopword
        # df_spark = df_spark.drop("description_no_stop")

        stopwords_list = StopWordsRemover.loadDefaultStopWords("english")
        stopwords_customize_list = ["app","apps"]
        stopwords_list = np.append(stopwords_list,stopwords_customize_list)

        stopwords = StopWordsRemover(inputCol="description_lemm",outputCol="description_no_stop",stopWords=stopwords_list)
        stopwords.getStopWords()
        df_spark = stopwords.transform(df_spark)


        df_pd_desc_final = df_spark.toPandas()

        # ### Note: IDF vector must be trained with large corpus, otherwise lose the advance of IDF


        # get the "description" column
        joinF= lambda x:" ".join(x)
        df_pd_desc_final["description_final"] = df_pd_desc_final["description_no_stop"].apply(joinF)

        corpus_list = df_pd_desc_final["description_final"].tolist()

        df_pd_desc_final = get_tfidf(corpus_list,df_pd_desc_final,self.topn)
        
        return df_pd_desc_final

    
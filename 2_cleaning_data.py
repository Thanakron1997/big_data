import pandas as pd
import spacy 
import re
from multiprocessing import cpu_count

num_processes = cpu_count() - 1


def set_rank(x):
    '''clean revivew data'''
    if x <= 2:
        return -1
    elif x == 3:
        return 0
    else:
        return 1
    

def cleaning(doc):
    '''Lemmatizes and removes stopwords'''
    txt = [token.lemma_ for token in doc if not token.is_stop]
    removelst = ['cq', 'rg', 'bb']
    txt = [w for w in txt if w not in removelst]
    if len(txt) > 2:
        return ' '.join(txt)

def clean_data():
    '''clean data process'''
    # import data
    df_tt = pd.read_csv("Twitter_Data.csv",low_memory=False)
    df_red = pd.read_csv("Reddit_Data.csv",low_memory=False)
    df_rw = pd.read_csv("reviews.csv",low_memory=False)

    # clean twitter data
    df_tt = df_tt.dropna().reset_index(drop=True)
    df_tt = df_tt.reset_index(drop=True)
    # clean Raddit data
    df_red = df_red.dropna().reset_index(drop=True)
    df_red = df_red.reset_index(drop=True)
    df_red = df_red.rename(columns={'clean_comment':'clean_text'})
        
    df_rw['category'] = df_rw['stars'].apply(set_rank)
    df_rw_cut = df_rw[['text','category']]
    df_rw_cut = df_rw_cut.dropna().reset_index(drop=True)
    df_rw_cut = df_rw_cut.rename(columns={'text':'clean_text'})
    df = pd.concat([df_tt,df_red,df_rw_cut]).reset_index(drop=True)
    df = df.drop_duplicates()
    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['clean_text'])
    nlp = spacy.load("en_core_web_lg",disable=['ner', 'parser'])
    spacy.prefer_gpu()
    df['clean'] = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, n_process=num_processes, batch_size=5000)]
    del nlp
    df = df.dropna().reset_index(drop=True)
    print(df.isnull().sum())
    df.to_csv("raw_data.csv",index=False)
    df2 = df[['clean','category']]
    df2.to_csv("raw_data_2.csv",index=False)

if __name__ == "__main__": 
    clean_data()

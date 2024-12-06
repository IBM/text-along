import spacy
import urllib.parse
import requests
from bs4 import BeautifulSoup
from spacytextblob.spacytextblob import SpacyTextBlob
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Load the spacy english models
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the semantic similarity models
print("Loading Similarity Model")
# self.model = SentenceTransformer('stsb-roberta-large')
model = SentenceTransformer('all-mpnet-base-v2', cache_folder='APP/eppbias/cache')
print("Done.")

# For formating our results dictionary to json without errors. 
def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
        
def sentiment_analysis(articles_text):
    """
    Given a dictionary of articles and their text, return the sentiment of each one
    and the average  

        param: articles_text, keys are tuples: (article title, article url), value is the text as a str; or just the text of the article
            type: dict or str

        returns:
            results, key is article's title, value is sentiment/polarity
                type: dict
            
            avg_sentiment
                type: float
    """
    if type(articles_text) is dict:
        results = dict()
        avg_sentiment = 0

        for article_info in articles_text:
            doc = nlp(articles_text[article_info])

            results[article_info[0]] = doc._.polarity

            avg_sentiment += doc._.polarity
        avg_sentiment = avg_sentiment / len(articles_text)

        return results, avg_sentiment
    else:
        doc = nlp(articles_text)

        return doc._.polarity

def type_checker(val, intended_type):
    """
    Type checker function, ensures that the 
    type of the value in question is of the intended_type,
    errors if not

    param: val
        type: any
    param: intended_type
        type: Python keyword denoting a data type (int, list, str...)
    returns: 
        type: None or TypeError
    """
    if type(val) is intended_type:
        return
    else:
        raise TypeError(f'{val} is of type {type(val)}, but it should be of type {intended_type}')

def articles_text_to_json(articles_text):
    """
    This function accepts an articles_text, which is a dictionary of this schema:
    keys are tuples: (article title, article url), value is the text.

    It returns this dictionary as a list of dictionaries of this schema:

        "article_url":
        "article_title":
        "article_text":
    param: articles_text
        type: dict

    returns: json_articles_text
        type: list of dict (will be transformed into JSON by user)
    """
    json_articles_text = []
    for article in articles_text:
        pre_json_obj = dict()
        pre_json_obj["article_url"] = article[1]
        pre_json_obj["article_title"] = article[0]
        pre_json_obj["article_text"] = articles_text[3]

        if len(article) == 3:
            pre_json_obj["outlet_name"] = article[2]

        json_articles_text.append(pre_json_obj)

    return json_articles_text

def getPoliticalView(newspaper: str):
    """
    Helper function from Luca Weissbeck, scrapes AllSides to find the political view of a given News outlet

    param: newspaper
        type: str

    returns: result if found
        type: str or None
    """
    URL = "https://www.allsides.com/media-bias/media-bias-ratings?field_featured_bias_rating_value=All&field_news_source_type_tid[1]=1&field_news_source_type_tid[2]=2&field_news_source_type_tid[3]=3&field_news_source_type_tid[4]=4&field_news_bias_nid_1[1]=1&field_news_bias_nid_1[2]=2&field_news_bias_nid_1[3]=3&title=" + urllib.parse.quote(newspaper)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("td", class_="views-field views-field-field-bias-image")
    if results: 
        result = results[0].find("img").get("title").split(":")[1]
        return result
    else:
        return None

def count(article_text, words_to_check):
    """
    A helper function that takes in the text of an article, 
    a list of words to look for in that text, and returns the
    total amount of article's words that also occur in the list 

    param: article_text
        type: str

    param: words_to_check
        type: list of str

    returns: count
        type: int
    """
    tokenized_text = article_text.split(" ")

    count = 0
    for word in tokenized_text:
        if word in words_to_check:
            count += 1

    return count

def get_averages(all_categories, categories_found, values_for_categories_found):
    """
    Given a list of possible categories that can appear, a list of categories that actually appear (the x's so to speak),
    and a list of values respective to the list of categories that actually appear (the y's), compute the averages
    for each possible category

    param: all_categories
        type: list

    param: categories_found
        type: list

    param: values_for_categories_found
        type: list of int

    returns: results
        type: list of int
    """
    results = []

    for category in all_categories:
        count_category = 0
        sum_values_for_category = 0
        for idx in range(len(categories_found)):
            if categories_found[idx] == category:
                count_category += 1
                sum_values_for_category += values_for_categories_found[idx]
        
        if count_category != 0:
            results.append(sum_values_for_category / count_category)
        else:
            results.append(0)
    return results

def semantic_search(sentence, corpus_of_sentences: list):
    '''
    Retrieve Top K most similar sentences from a corpus given a sentence.
    '''
    corpus_embedding = model.encode(corpus_of_sentences, convert_to_tensor=True)
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    top_k=len(corpus_of_sentences)
    cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embedding)[0]
    top_results = np.argpartition(-cos_scores.to(device), range(top_k))[0:top_k]
    ranked_output = {"sentences":[], "distance_scores":[]}
    for idx in top_results[0:top_k]:
        ranked_output['sentences'].append(corpus_of_sentences[idx])
        ranked_output['distance_scores'].append("%.4f" % (cos_scores[idx]))
    return ranked_output

def get_word_semantic_similarity(word1, word2):
    ''' 
    Find semantic similarity between two words.
    '''
    embedding1 = model.encode(word1, convert_to_tensor=True)
    embedding2 = model.encode(word2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    similarity_score = cosine_scores.item()
    return similarity_score
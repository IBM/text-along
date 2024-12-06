# Give python access to content in root directory
from utils.utilities import count
from time import sleep
from typing import Union
import numpy as np
from math import e, sqrt
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from tmi.tmi_utils import Helperfunctions, DescriptorAnalysis
from tmi.tmi_errors import DataframeMissingColumnsError
import string
import re
import pandas as pd
import pickle
from nltk import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import logging
logging.basicConfig(
    filename='tmi_logic.log',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO
)
import requests
from tmi.tmi_errors import ApiNotStarted

class TMIAnalysis:
    """
    This class contains multiple methods to perfrom an analysis in order to find out
    whether or not too much information is given about a specific subject.
    All methods can be invoked indepedently from each other and manipulate the class's dataframe.
    """

    def __init__(self, article_dictionary: dict=None, remove_duplicates=True, df=None, corenlp_url="https://127.0.0.1:9000") -> None:
        """
        Initiliases the dataframe by adding the columns "URL", "article_content", "headline" and "news_outlet".
        All of the columns are extracted  from the article_dictionary.

        :param article_dictionary: Article dictionary retrieved directly from BingAPI method "get_articles". MUST contain at least TWO entries.
        :type article_dictionary: dict
        """
        # Setup dataframe
        self.dataframe = df
        # Check if Java CoreNLP Server is started
        if requests.get(corenlp_url).status_code != 200:
            raise ApiNotStarted("The corenlp server is required but not running. Kindly start the server")
        else:
            self._corenlp_url = corenlp_url

        if self.dataframe is None:
            self.__create_corpus(article_dictionary,
                                remove_duplicates=remove_duplicates)
            self.__clean_article_text()
            self.hf = Helperfunctions()

    def __create_corpus(self, articles: list, remove_duplicates: bool = True) -> None:
        """
        Article dictionary retrieved directly from BingAPI method "get_articles".

        :param articles: Manipulates dataframe by filling it with headline, URL & newsoutlet name.
        :type articles: dict
        """
        headline = list()
        URL = list()
        newsoutlet_name = list()
        dataframe_dictionary = dict()
        article_content = list()

        for item in articles:
            if len(item) < 4 or type(item) == string:
                continue

            if len(item[3]) < 5:
                continue
            headline.append(item[0])
            URL.append(item[1])
            newsoutlet_name.append(item[2])
            article_content.append(item[3])

        # Create dictionary as basis for Dataframe
        dataframe_dictionary = {"URL": URL, "headline": headline,
                                "newsoutlet_name": newsoutlet_name, "article_content": article_content}
        dataframe = pd.DataFrame.from_dict(dataframe_dictionary)
        dataframe.set_index("URL", inplace=True)
        if remove_duplicates:
            dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
        self.dataframe = dataframe

    def filter_out_articles(self, category: str) -> None:
        """
        Filter out all articles that are not recognised as the category string by Watson.
        Applicable categories can be found at: https://cloud.ibm.com/docs/natural-language-understanding?topic=natural-language-understanding-categories-hierarchy-v1
        E.g. "/society/crime"

        :param category: Category that should NOT be filtered out.
        :type category: str
        """
        self.hf = Helperfunctions()
        counter = 0
        for i, row in self.dataframe.iterrows():
            if counter < 8700:
                counter += 1
                continue
            if counter % 20 == 0:
                print(counter)

            try:

                result_dict = self.hf.get_NLU_categories(i)
                for item in result_dict["categories"]:
                    #self.dataframe.loc[i]["entities"] = result_dict["entities"]
                    if (category not in item["label"]) and (result_dict["categories"].index(item) == len(result_dict["categories"]) - 1):
                        self.dataframe.drop(i, inplace=True)
                        print("removed", i)
                    elif category in item["label"]:
                        break
                    else:
                        continue

            # This might occur, when Watson can not fetch the URL --> we skip the URL hoping the topic applies
            except Exception as e:
                print(e)
                print(i)
                continue

            finally:
                counter += 1

    def get_entities(self) -> None:
        """
        Get the entities in an article from Watson NLU.
        """
        self.dataframe["entities"] = ""
        for i, row in self.dataframe.iterrows():
            try:
                result_dict = self.hf.get_NLU_categories(i)
                entities = list()
                for item in result_dict["entities"]:
                    entities.append(
                        {"type": item["type"], "text": item["text"]})

                self.dataframe.loc[i]["entities"] = entities

            # This might occur, when Watson can not fetch the URL --> we skip the URL hoping the topic applies
            except Exception as e:
                print(e)
                continue

    def __clean_article_text(self) -> None:
        """
        Cleans the article text of any characters that might interefere with any followin analysis.
        """
        def clean3(x): return re.sub('\w*\d\w*', '', x)
        def clean4(x): return re.sub('[‘’“”…]', '', x)

        self.dataframe["article_content"] = self.dataframe["article_content"].apply(
            clean3).apply(clean4)

    def clean_text(self, text:str) -> None:
        """
        Cleans the text of any characters that might interefere with any analysis.
        """
        cleaned_text = re.sub('[^a-zA-Z0-9 \n\.]', '', text)
        return cleaned_text


    def find_ethnicity_keywords(self, port: int = 9000) -> None:
        """
        Adds count of specific ethnic keywords in article text as a new column.
        """
        # Load keywords
        dirname = os.path.dirname(__file__)
        ethnicity_keyords = os.path.join(
            dirname, "../../../data/lexicons/ethnicity-keywords.pkl")
        keywords = pickle.load(
            open(ethnicity_keyords, "rb"))
        keywords = list(map(lambda x: x.lower(), keywords))
        ps = PorterStemmer()
        keywords = list(map(lambda x: ps.stem(x), keywords))

        # Updated approach

        keywords_lists = list()
        for i, row in self.dataframe.iterrows():
            try:
                sentences_in_article = sent_tokenize(
                    self.dataframe.loc[i]["article_content"])
            except TypeError:
                keywords_lists.append("")
                continue
            filtered_valid_ethnic_keywords_article = dict()
            for sentence in sentences_in_article:
                words_in_sentence = sentence.split()
                words_in_sentence_stemmed = list(
                    map(lambda x: ps.stem(x), words_in_sentence))
                word_in_sentence_translator = dict(
                    zip(words_in_sentence_stemmed, words_in_sentence))
                keywords_in_sentence = list(
                    set(keywords) & set(words_in_sentence_stemmed))
                entitymentions = self.hf.get_all_entitymentions(sentence, 9000)
                if entitymentions is None:
                    continue

                for keyword in keywords_in_sentence:
                    word_valid = True
                    original_word = word_in_sentence_translator[keyword]

                    for entity_dict in entitymentions:
                        if original_word in entity_dict["text"] and entity_dict["ner"] != "NATIONALITY":
                            word_valid = False

                    if word_valid and keyword not in filtered_valid_ethnic_keywords_article:
                        filtered_valid_ethnic_keywords_article[keyword] = 1

                    elif word_valid and keyword in filtered_valid_ethnic_keywords_article:
                        filtered_valid_ethnic_keywords_article[keyword] += 1

            keywords_lists.append(filtered_valid_ethnic_keywords_article)

        self.dataframe["ethnicities"] = keywords_lists

    def __get_headline_body_length(self) -> None:
        """
        Calculates the word length of headline & article text.
        """
        headline_length = list()
        article_length = list()
        for i, row in self.dataframe.iterrows():
            headline_length.append(int(len(self.dataframe.loc[i]["headline"])))
            article_length.append(
                int(len(self.dataframe.loc[i]["article_content"])))
        self.dataframe["headline_length"] = headline_length
        self.dataframe["article_length"] = article_length

    def article_length_mediatype_correlation(self, article_types: list) -> Union[list, list, list]:
        """
        Determine the relationship between the type of newspaper (print/online media and online only media)
        and the overall article length. The returned arguments can be used for plotting.

        :param article_types: An ordered list of strings containing the type of news_outlet for the corresponding row.
                              String is either "online" or "print".
        :type article_types: list
        :return: Returns three lists. 1: X-Axis Values 2: Standard error for online & print, 3: Averages of word length for online & print
        :rtype: Union[list, list, list]
        """
        # Prerequisites
        if not pd.Series(["headline_length", "article_length"]).isin(self.dataframe.columns).all():
            self.__get_headline_body_length()

        # Adding the articles_typed list as column
        self.dataframe["newspaper_type"] = article_types

        averages = [self.dataframe[self.dataframe["newspaper_type"] == "print"].article_length.mean(
            axis=0), self.dataframe[self.dataframe["newspaper_type"] == "online"].article_length.mean(axis=0)]
        x = ["Print/Online", "Online"]

        # Adding Mean Error to graph
        print_stdev = np.std(
            self.dataframe[self.dataframe["newspaper_type"] == "print"].article_length.values)
        online_stdev = np.std(
            self.dataframe[self.dataframe["newspaper_type"] == "online"].article_length.values)
        print_sterr = print_stdev / \
            sqrt(len(
                self.dataframe[self.dataframe["newspaper_type"] == "print"].article_length.values))
        online_sterr = online_stdev / \
            sqrt(len(
                self.dataframe[self.dataframe["newspaper_type"] == "online"].article_length.values))
        error = [print_sterr, online_sterr]

        return x, error, averages

    def extract_subject_object_action(self) -> None:
        """
        Extracting the subject, object and action from the headline using Watson NLP.
        """
        HL_Subject = list()
        HL_Object = list()
        HL_Actions = list()

        for i, row in self.dataframe.iterrows():
            HL_Subject_temp = []
            HL_Object_temp = []
            HL_Actions_temp = []

            try:
                semantic_roles_dict = self.hf.get_NLU_semanticroles(
                    self.dataframe.loc[i]["headline"])
                for item in semantic_roles_dict:
                    if "subject" in item:
                        HL_Subject_temp.append(item["subject"]["text"])
                    if "object" in item:
                        HL_Object_temp.append(item["object"]["text"])
                    if "action" in item:
                        HL_Actions_temp.append(item["action"]["text"])

            # Could be thrown if text includes foreign language
            except Exception as e:
                print(e)
                continue

            finally:
                HL_Subject.append(HL_Subject_temp)
                HL_Object.append(HL_Object_temp)
                HL_Actions.append(HL_Actions_temp)
            # Wait 1 second otherwise API Exception: Too many requests
            sleep(1)
        self.dataframe["HL_Subject"] = HL_Subject
        self.dataframe["HL_Action"] = HL_Actions
        self.dataframe["HL_Object"] = HL_Object

    def is_subject_in_headline(self) -> None:
        """
        Determines for each row whether a subject is present in the corresponding headline.
        """
        # Prerequisites
        if not pd.Series(["HL_Subject", "HL_Action", "HL_Object"]).isin(self.dataframe.columns).all():
            self.extract_subject_object_action()

        has_subject = list()

        for i, row in self.dataframe.iterrows():
            if len(self.dataframe.loc[i]["HL_Subject"]) > 0:
                has_subject.append(True)
            else:
                has_subject.append(False)

        self.dataframe["is_subject_in_HL"] = has_subject

    def object_correspond_context(self, context: str) -> None:
        """
        Does the object in the headline equal the context of the headline?
        E.g. "Daycare Worker charged with murder after assaulting baby who wouldn't nap, cops say"
        The object (murder) == context (murder)
        An object not corresponding to the context might be a sign of framing bias. E.g. the object
        is the age of the murderer.

        :param context: The context for the analysis. Usually aligns with the keyword searched for
                        with BingAPI.
        :type context: str
        """
        if not pd.Series(["HL_Subject", "HL_Action", "HL_Object"]).isin(self.dataframe.columns).all():
            self.extract_subject_object_action()

        corresponds_to_context = list()

        for i, row in self.dataframe.iterrows():
            # print(i)

            try:
                corresponds_to_context_row = False
                for hl_object in self.dataframe.loc[i]["HL_Object"]:
                    # A further check for synonyms / autonyms might be appropriate here
                    try:
                        if context in hl_object.lower():
                            corresponds_to_context_row = True
                    except Exception as e:
                        print(e)
                        continue
                corresponds_to_context.append(corresponds_to_context_row)

            except Exception as e:
                print(e)
                corresponds_to_context_row = False

        self.dataframe["object_correspond_context_HL"] = corresponds_to_context

    def is_subject_entity_person(self) -> None:
        """
        Is the subject mentioned in the headline an entity of type person?
        Determined using Watson NLP's entities.
        """
        # Prerequisites
        if not pd.Series(["HL_Subject", "HL_Action", "HL_Object"]).isin(self.dataframe.columns).all():
            self.extract_subject_object_action()

        if not pd.Series(["entities"]).isin(self.dataframe.columns).all():
            self.get_entities()

        entity_and_type_person = list()

        for i, row in self.dataframe.iterrows():
            try:
                entity_and_type_person_row = False
                for entity in self.dataframe.loc[i]["entities"]:
                    if entity["type"] == "Person" and entity["text"] in " ".join(self.dataframe.loc[i]["HL_Subject"]):
                        entity_and_type_person_row = True

            except Exception as e:
                print(e)
                entity_and_type_person_row = False
                continue

            entity_and_type_person.append(entity_and_type_person_row)

        self.dataframe["entity_and_person"] = entity_and_type_person

    def political_spectrum(self) -> None:
        """
        Determine the political spectrum of the specific newspaper outlet using information provided by the
        website "www.allsides.com".
        """
        political_spectrum = lambda x: self.hf.get_political_spectrum(x)

        self.dataframe["political_spectrum"] = self.dataframe["newsoutlet_name"].apply(
            political_spectrum)

    def hedges_booster_factiveverbs(self) -> None:
        """
        Adds hedges,  boosters and factive verb counts as individual columns to the dataframe.
        """
        dirname = os.path.dirname(__file__)
        boosters_path = os.path.join(
            dirname, "../../../data/lexicons/boosters.pickle")
        boosters = pickle.load(open(boosters_path, "rb"))
        hedges_path = os.path.join(
            dirname, "../../../data/lexicons/hedges.pickle")
        hedges = pickle.load(open(hedges_path, "rb"))
        factive_verbs_path = os.path.join(
            dirname, "../../../data/lexicons/factive_verbs.pickle")
        factive_verbs = pickle.load(open(factive_verbs_path, "rb"))
        bias_lexicon_path = os.path.join(
            dirname, "../../../data/lexicons/bias_lexicon.pickle")
        bias_lexicon = pickle.load(open(bias_lexicon_path, "rb"))

        boosters_count = list()
        hedges_count = list()
        factive_verbs_count = list()
        bias_lexicon_count = list()

        for i, row in self.dataframe.iterrows():
            boosters_count.append(
                count(self.dataframe.loc[i]["article_content"], boosters))
            hedges_count.append(
                count(self.dataframe.loc[i]["article_content"], hedges))
            factive_verbs_count.append(
                count(self.dataframe.loc[i]["article_content"], factive_verbs))
            bias_lexicon_count.append(
                count(self.dataframe.loc[i]["article_content"], bias_lexicon))

        self.dataframe["boosters_count"] = boosters_count
        self.dataframe["hedges_count"] = hedges_count
        self.dataframe["factive_verbs_count"] = factive_verbs_count
        self.dataframe["bias_lexicon_count"] = bias_lexicon_count

    def get_img_caption_byline_excerpt(self, api_port: int = 8080) -> None:
        """
         Retrieves the image caption (if available), byline and excerpt from the article.
        !Important!: NodeJS API needs to be started locally. (Located in "./readability-api")

        :param api_port: Port the NodeJS API is running on, defaults to 8080
        :type api_port: int, optional
        """
        try:
            image_captions = list()
            bylines = list()
            excerpts = list()

            def clean3(x):
                if x is None:
                    return x
                if type(x) == list:
                    return_list = list()
                    for item in x:
                        if len(x) > 1:
                            return_list.append(re.sub('\w*\d\w*', '', item))
                    return return_list
                else:
                    return re.sub('\w*\d\w*', '', x)

            def clean4(x):
                if x is None:
                    return x
                return re.sub('[‘’“”…]', '', x)

            for i, row in self.dataframe.iterrows():
                try:
                    request = self.hf.get_image_tags(i, api_port)
                    image_captions.append(request[0])
                    bylines.append(request[1])
                    excerpts.append(request[2])

                except Exception as e:
                    print(e)
                    image_captions.append("")
                    bylines.append("")
                    excerpts.append("")
                    continue

            self.dataframe["image_captions"] = image_captions
            self.dataframe["byline"] = bylines
            self.dataframe["excerpts"] = excerpts

           # self.dataframe["image_captions"] = self.dataframe["image_captions"].apply(clean3).apply(clean4)
            self.dataframe["byline"] = self.dataframe["byline"].apply(
                clean3).apply(clean4)
            self.dataframe["excerpts"] = self.dataframe["excerpts"].apply(
                clean3).apply(clean4)

        except Exception as e:
            print(e)

    def descriptors_headline(self) -> None:
        """
        Adds the descriptor count and name as columns to dataframe. Classified as descriptor is any node left
        on a "NN" node that is also a parent of the same "NP" node. Exceptions include english stopwords.

        :param port: Port the CoreNLP server is running on, defaults to 9000
        :type port: int, optional
        """
        descriptor_count = list()
        descriptor_name = list()
        logging.info("GETTING descriptors counts......")
        for i, _ in self.dataframe.iterrows():
            descriptor_analysis = DescriptorAnalysis(self._corenlp_url)
            sentence = self.dataframe.loc[i]["headline"]
            # the tree parse won't work on a very long sentence. so we just fix the values
            if len(sentence) <= 350:
                tuple_response = descriptor_analysis.get_descriptor_count(self.clean_text(sentence))
                descriptor_count.append(tuple_response[0])
                descriptor_name.append(tuple_response[1])
            else:
                descriptor_count.append(5)
                descriptor_name.append(['long sentence'])

        self.dataframe["descriptor_count"] = descriptor_count
        self.dataframe["descriptor_name"] = descriptor_name
        logging.info("FINISHED getting descriptor counts.......")

    def get_descriptor_sentiment(self, descriptor_count: int) -> str:
        """
        Determines the sentiment of descriptors solely based on the count of descriptors in the healdine.

        :param descriptor_count: Count of descriptors in headline.
        :type descriptor_count: int
        :return: Returns any sentiment from ["Positive", "Neutral", "Negative"]
        :rtype: str
        """
        if descriptor_count < 2:
            return "Positive"
        elif descriptor_count == 2:
            return "Neutral"
        else:
            return "Negative"

    def is_biased_from_descriptor(self) -> None:
        """
        Determines whether or not headline is biased based from count of descriptors. 
        Rules: < 2 descriptors  -> Positive (no tmi)
               == 2 descriptors -> Neutral (no tmi)
               > 2 descriptors  -> Negative (tmi) 
        """
        # Prerequisites:
        logging.info("START is_biased_from_descriptor......")
        if not pd.Series(["descriptor_count", "descriptor_name"]).isin(self.dataframe.columns).all():
            self.descriptors_headline()

        is_biased = lambda x: "tmi" if x > 2 else "no tmi"

        self.dataframe["descriptor_sentiment"] = self.dataframe["descriptor_count"].apply(
            self.get_descriptor_sentiment)
        self.dataframe["descriptor_class"] = self.dataframe["descriptor_count"].apply(
            is_biased)
        logging.info(f"FINISHED is_biased_from_descriptor...... {self.dataframe}")
        return self.dataframe

    def account_for_positively_biased(self, port: int = 9000) -> None:
        """
        The established bias from the function "is_biased_from_descriptor" automatically
        labels every headline having more than 2 descriptors as Negatively biased. 
        However, there might be cases where a headline has more than 2 descriptors but
        is positively biased. To distinguish this case, a CoreNLP sentiment analysis
        is performed on any sentence containing more than 2 descriptors. 

        :param port: Port the CoreNLP server is running on, defaults to 9000
        :type port: int, optional
        """
        # Prerequisites
        if not pd.Series(["descriptor_sentiment", "descriptor_class"]).isin(self.dataframe.columns).all():
            self.is_biased_from_descriptor()

        CoreNLP_sentiment_rating = {
            0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

        number_to_rating = lambda x: CoreNLP_sentiment_rating[int(x)]

        is_postively_biased = lambda x: "Positively Biased" if self.hf.corenlp_sentiment_analysis(
            x, port) > 2 else "Negatively Biased"

        self.dataframe["headline_sentiment"] = self.dataframe["headline"].apply(
            self.hf.corenlp_sentiment_analysis, port=port).apply(number_to_rating)
        self.dataframe["descriptor_class"][self.dataframe["descriptor_class"] ==
                                           "Negative"] = self.dataframe["headline"][self.dataframe["descriptor_class"] == "Negative"].apply(is_postively_biased)

    def object_subject_mention_imagecaptions(self) -> None:
        """
        Check whether the subject / object or both are mentioned in the headline.
        Adds new column with S = subject in headline, O = object in headline, S/O = Subject&Object in headline.
        """
        # Prerequisites
        if not pd.Series(["image_captions", "byline", "excerpts"]).isin(self.dataframe.columns).all():
            self.get_img_caption_byline_excerpt()

        if not pd.Series(["HL_Subject", "HL_Action", "HL_Object"]).isin(self.dataframe.columns).all():
            self.extract_subject_object_action()

        lowercase = lambda x: list(map(lambda y: y.lower(), x))
        replace_None = lambda x: list(
            map(lambda y: "No info" if y == None else y, x))
        # Replacing None values
        self.dataframe["image_captions"] = self.dataframe["image_captions"].apply(
            replace_None)

        # Converting everything to lowercase
        self.dataframe["HL_Object"] = self.dataframe["HL_Object"].apply(
            lowercase)
        self.dataframe["HL_Subject"] = self.dataframe["HL_Subject"].apply(
            lowercase)
        self.dataframe["image_captions"] = self.dataframe["image_captions"].apply(
            lowercase)

        image_caption_contains = list()

        for i, row in self.dataframe.iterrows():
            def tokenizer(list_of_sentences: str) -> list():
                temp_list = list()
                ps = PorterStemmer()
                for sentence in list_of_sentences:
                    temp_list += word_tokenize(sentence)
                    for index, item in enumerate(temp_list):
                        temp_list[index] = ps.stem(item)
                return temp_list

            objects_row = tokenizer(self.dataframe.loc[i]["HL_Object"])
            subjects_row = tokenizer(self.dataframe.loc[i]["HL_Subject"])
            image_captions_row = tokenizer(
                self.dataframe.loc[i]["image_captions"])

            objects = any(item in objects_row
                          for item in image_captions_row)
            subjects = any(item in subjects_row
                           for item in image_captions_row)

            if objects and subjects > 0:
                image_caption_contains.append("S/O")
            elif objects and not subjects:
                image_caption_contains.append("Object")
            elif subjects and not objects:
                image_caption_contains.append("Subject")
            else:
                image_caption_contains.append("Neither")

        self.dataframe["Subject/Object_in_img_caption"] = image_caption_contains
    """
    def image_attribute_analysis(self) -> None:
        
        #Analyses the images in article and identifies race & gender if a person is present.
        #Adds two columns to dataframes (image_src, image_gender).
        
        # Prerequisites
        if not pd.Series(["image_captions", "byline", "excerpts"]).isin(self.dataframe.columns).all():
            self.get_img_caption_byline_excerpt()

        race = list()
        gender = list()
        for i, row in self.dataframe.iterrows():
            temp_race = list()
            temp_gender = list()

            if len(self.dataframe.loc[i]["image_src"]) > 0:
                for image in self.dataframe.loc[i]["image_src"]:
                    analysis = self.hf.image_attribute_analysis(image)
                    if "dominant_race" in analysis and "gender" in analysis:
                        temp_race.append(analysis["dominant_race"])
                        temp_gender.append(analysis["gender"])
                    else:
                        temp_race.append("")
                        temp_gender.append("")

                if len(temp_race) or len(temp_gender) > 0:
                    race.append(temp_race)
                    gender.append(temp_gender)

                else:
                    race.append(temp_race[0])
                    gender.append(temp_gender[0])

                    sleep(10)
            else:
                race.append("")
                gender.append("")

        self.dataframe["image_race"] = race
        self.dataframe["image_gender"] = gender
    """
    def entity_action_relation(self, port: int = 9000) -> None:
        """
        Analysis of relations in article content. 
        Relation is made up of Subject -> Event -> Object

        :param port: Port of Java CoreNLP server, defaults to 9000
        :type port: int, optional
        """
        relation_strings_total = list()

        for i, row in self.dataframe.iterrows():
            relation_strings = list()
            result = self.hf.get_entity_action_relation(
                self.dataframe.loc[i]["article_content"], port)
            subjects = self.hf.get_all_subjects(
                self.dataframe.loc[i]["article_content"], port)
            if len(result) > 0:
                for item in result:
                    for relation in item:
                        # print(relation)
                        relation_string = relation["subject"] + "->" + \
                            relation["relation"] + "->" + relation["object"]
                        if relation_string not in relation_strings and relation["subject"] in subjects:
                            relation_strings.append(relation_string)

                relation_strings_total.append(relation_strings)

            else:
                relation_strings_total.append("")

        self.dataframe["event_relations"] = relation_strings_total

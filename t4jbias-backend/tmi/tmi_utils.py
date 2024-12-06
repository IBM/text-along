# Give python access to content in root directory
from urllib import request
from dotenv import load_dotenv
import os
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions, EntitiesOptions, SemanticRolesOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
from typing import Union
from tmi.tmi_errors import ApiNotStarted
import socket
from nltk.tree import Tree
from nltk.parse import CoreNLPParser
import nltk
nltk.download('stopwords', download_dir='./data/nltk_data')
from nltk.corpus import stopwords
from pycorenlp import StanfordCoreNLP
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

import logging
logging.getLogger(__name__)

class Helperfunctions:
    """
    Contains general tmi_utils for performing a TMI Analysis.
    """

    def __init__(self) -> None:
        """
        Initliases the Watson NLP service. 
        """
        dirname = os.path.dirname(__file__)
        env_path = os.path.join(dirname, "../../../.env")
        load_dotenv(dotenv_path=env_path, override=True)

        NLU_API_KEY = os.environ.get("NLU_API_KEY")
        NLU_SERVICE_URL = os.environ.get("NLU_SERVICE_URL")

        authenticator = IAMAuthenticator(NLU_API_KEY)
        self.natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2021-08-09',
            authenticator=authenticator
        )
        self.natural_language_understanding.set_service_url(NLU_SERVICE_URL)

    def get_NLU_categories(self, url_newspaper: str) -> dict:
        """
        Classifies newspaper article into predefined categories by Watson NLU.

        :param url_newspaper: URL of newspaper article.
        :type url_newspaper: str
        :return: Dict containing any classified category.
        :rtype: dict
        """
        return self.natural_language_understanding.analyze(
            url=url_newspaper,
            features=Features(categories=CategoriesOptions(limit=3),
                              entities=EntitiesOptions(sentiment=True, limit=5)
                              )).get_result()

    def get_NLU_semanticroles(self, headline_newspaper: str) -> dict:
        """
        Retrieves the semantic roles for a newspaper article from Watson NLU.

        :param headline_newspaper: Headline for the newspaper to be analysed. 
        :type headline_newspaper: str
        :return: Retrieved dictionary that contains any semantic roles in the headline.
        :rtype: dict
        """

        result = self.natural_language_understanding.analyze(
            text=headline_newspaper,
            features=Features(semantic_roles=SemanticRolesOptions())).get_result()

        return result["semantic_roles"]

    def get_political_spectrum(self, news_outlet: str) -> str:
        """
        Web crawler for retrieving political spectrum of a given newspaper. 

        :param news_outlet:  Name of the newspaper.
        :type news_outlet: str
        :return: String of political direction. Possible returns: [Left, Lean Left, Center, Lean Right, Right]
        :rtype: str
        """
        URL = "https://www.allsides.com/media-bias/media-bias-ratings?field_featured_bias_rating_value=All&field_news_source_type_tid[1]=1&field_news_source_type_tid[2]=2&field_news_source_type_tid[3]=3&field_news_source_type_tid[4]=4&field_news_bias_nid_1[1]=1&field_news_bias_nid_1[2]=2&field_news_bias_nid_1[3]=3&title=" + urllib.parse.quote(
            news_outlet)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find_all(
            "td", class_="views-field views-field-field-bias-image")
        if results:
            result = results[0].find("img").get("title").split(":")[1]
            return result
        else:
            return None

    def get_image_tags(self, url: str, port: int) -> Union[list, str, str]:
        """
        Retrieve the image tags, byline and excerpt from an article.
        This is done by calling a NodeJS Api which utilises the "readibility" package
        published by Mozilla Firefox. This clears the article's HTML of any advertisements that
        might mislead the analysis. 

        :param url: URL of the newspaper article.
        :type url: str
        :param port: Port that NodeJS API is running on.
        :type port: int
        :raises ApiNotStarted: Raised if user forgot to start the NodeJS API.
        :return: Returns the list of image captions retrieved as well as the byline and excerpt.
        :rtype: Union[list, str, str]
        """
        # Checking if API is running
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_check = a_socket.connect_ex(("127.0.0.1", port))
        a_socket.close()
        if port_check != 0:
            raise ApiNotStarted(
                "API is most likely not started. If you have changed the API port make sure to call get_img_caption_byline_excerpt with the optional port parameter.")

        page = requests.get(url)
        filtered_article = requests.post(
            "http://localhost:" + str(port) + "/filter", data={"text": str(page.content), "url": url})
        filtered_article = json.loads(
            filtered_article.content.decode("utf-8"))
        filtered_article["doc"]
        results = BeautifulSoup(
            filtered_article["doc"], features="lxml").findAll("img")

        image_tags = []
        for i in range(len(results)):
            image_tags.append(results[i].get("alt"))

        if "byline" not in filtered_article:
            filtered_article["byline"] = ""

        if "excerpt" not in filtered_article:
            filtered_article["excerpt"] = ""

        return image_tags, filtered_article["byline"], filtered_article["excerpt"]

    def return_nltk_tree(self, sentence: str, port: int = 9000) -> Tree:
        """
        Return the NLTK tree of sentence.

        :param sentence: Sentence to retrieve NLTK tree from.
        :type sentence: str
        :param port: Port CoreNLP server is running on, defaults to 9000
        :type port: int, optional
        :raises ApiNotStarted: Raised if you user forgot to start the server.
        :return: NLTK tree of sentence.
        :rtype: Tree
        """
        # Check if Java CoreNLP Server is started
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_check = a_socket.connect_ex(("127.0.0.1", port))
        a_socket.close()

        if port_check != 0:
            raise ApiNotStarted(
                "API is most likely not started. If you have changed the API port make sure to call API is most likely not started. If you have changed the API port make sure to call get_nltk_tree with the optional port parameter.")

        parser = CoreNLPParser(url='http://localhost:9000')
        root = list(parser.raw_parse(sentence))[0][0]
        return root

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

    def corenlp_sentiment_analysis(self, headline: str, port: int = 9000) -> int:
        """
        Retrieve the overall sentiment of a sentence via CoreNLP.

        :param headline: Sentence or headline to be analysed.
        :type headline: str
        :param port: Port on which CoreNLP server is running on, defaults to 9000.
        :type port: int, optional
        :raises ApiNotStarted: Raised if user forgot to start CoreNLP server.
        :return: Returns the score for sentiment ranging from 0(Very negative) to 4 (Very positive).
        :rtype: int
        """
        # Check if Java CoreNLP Server is started
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_check = a_socket.connect_ex(("127.0.0.1", port))
        a_socket.close()

        if port_check != 0:
            raise ApiNotStarted(
                "API is most likely not started. If you have changed the API port make sure to call API is most likely not started. If you have changed the API port make sure to call account_for_positvely_biased with the optional port parameter.")

        nlp = StanfordCoreNLP("http://localhost:9000")

        result = nlp.annotate(headline, properties={
                              'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 1000})

        if type(result) == str:
            result = json.loads(result)
        for s in result["sentences"]:
            return s["sentimentValue"]

    # def __delete_image_locally(self, filename: str) -> None:
        """
        Deletes the previous downloaded image.

        :param filename: Filename of the image.
        :type filename: str
        :raises FileNotFoundError: If file to delete is not found.
        """
        """
        if os.path.exists(filename):
            os.remove(filename)
        else
            raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename)
        """
   # def image_attribute_analysis(self, image_url: str) -> dict:
        """
        Performs an image attribute analysis on the image using DeepFace.

        :param image_url: Online URL of image to perform analysis on.
        :type image_url: str
        :return: Contains analysis results. (dominant race, gender, dominant emotion, age)
        :rtype: dict
        """
        """
        # Save image locally
        try:
            filename = str(uuid.uuid4())
            filepath = wget.download(image_url, out=filename)
        except:
            return {}
        
        try:
            analysis = DeepFace.analyze(filepath, actions=["age", "gender", "race", "emotion"])
            self.__delete_image_locally(filepath)
        except:
            print("No Face Detected")
            self.__delete_image_locally(filepath)
            analysis =  {}
    
        return analysis
        """

    def get_entity_action_relation(self, text: str, port: int = 9000) -> list:
        """
        Retrieves all relations of Subject -> Event -> Object.

        :param text: Text relations should be extracted from.
        :type text: str
        :param port: Port the CoreNLP Java Server is running on, defaults to 9000.
        :type port: int, optional
        :raises ApiNotStarted: Raised if CoreNLP Java Server has not been started.
        :return: Returns a list where each item represents the OpenIE analysis for one sentence. 
        :rtype: list
        """
        # Check if Java CoreNLP Server is started
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_check = a_socket.connect_ex(("127.0.0.1", port))
        a_socket.close()

        if port_check != 0:
            raise ApiNotStarted(
                "API is most likely not started. If you have changed the API port make sure to call API is most likely not started. If you have changed the API port make sure to call the function with the correct port parameter.")

        nlp = StanfordCoreNLP("http://localhost:" + str(port))

        output = nlp.annotate(text, properties={"annotators": "tokenize,ssplit,pos,depparse,natlog,openie",
                                                "outputFormat": "json", "triple.strict": "true"})

        result = list()
        try:
            for item in output["sentences"]:
                if "openie" in item:
                    result.append(item["openie"])
        except TypeError:
            result = []
            return result
        return result

    def get_all_subjects(self, text: str, port: int = 9000) -> None:
        """
        Returns all subjects for a given text.

        :param text: Text from which subjects should be extracted.
        :type text: str
        :param port: Port the CoreNLP Java server is running on, defaults to 9000.
        :type port: int
        """
        nlp = StanfordCoreNLP("http://localhost:" + str(port))
        output = nlp.annotate(text, properties={"annotators": "ner",
                                                "outputFormat": "json"})
        # print(output["sentences"])
        subjects = list()
        try:
            for sentence in output["sentences"]:
                for entitymention in sentence["entitymentions"]:
                    if entitymention["ner"] == "PERSON" or entitymention["ner"] == "ORGANIZATION":
                        subjects.append(entitymention["text"])

        except Exception as e:
            print(e)

        subjects = set(subjects)
        subjects = list(subjects)
        return subjects

    def get_all_entitymentions(self, sentence: str, port: int = 9000) -> list:
        """
        Retrives all recongised entities in sentence by CoreNLP.

        :param sentence: Sentence to extract entities from.
        :type sentence: str
        :param port: Port the CoreNLP server is running on, defaults to 9000.
        :type port: int
        :return: List of dictionaries each concerning a single entity mention.
        :rtype: list
        """

        nlp = StanfordCoreNLP("http://localhost:" + str(port))
        output = nlp.annotate(sentence, properties={"annotators": "ner",
                                                    "outputFormat": "json"})

        try:
            return output["sentences"][0]["entitymentions"]

        except Exception as e:
            print(e)


class DescriptorAnalysis:
    """
    Performing a Descriptor Analysis. 
    """

    def __init__(self, corelnp_url='http://127.0.0.1:9000') -> None:
        """
        Initiliases the counter for keeping track of the descriptor count in a sentence.
        Also initiliases the list to which every found descriptor is appended.
        """
        self.counter = 0
        self.descriptor_list = list()
        self._corelnp_url = corelnp_url

    def __exists_np_node(self, root: list) -> Union[bool, int]:
        """
        Checks whether or not an NP node exists in the given tree. 

        :param root: Part of the tree to search.
        :type root: list
        :return: The boolean describes whether or not an NP node exists. 
                 The integer describes where at what index of the list the NP node is located. Defaults to 0 if no NP node present.
        :rtype: Union[bool, int]
        """
        # There can be no descriptor if only one node
        # The root height is set to 4 as another subtree of NML nodes can exist
        if len(root) == 1 or root.height() > 4:
            return False, 0

        # Save position of NN and replace with far rightest NN
        far_right_NN = [False, 0]

        for i in range(1, len(list(root))):
            if root[i].label() == "NN":
                far_right_NN[0], far_right_NN[1] = True, i

        return far_right_NN[0], far_right_NN[1]

    def __traverse(self, root: Tree) -> None:
        """
        Traversal algorithm for searching a NLTK Tree. 

        :param root: NLTK Tree to search.
        :type root: Tree
        """
        try:
            root.label()

        except AttributeError:
            return
        else:
            # Check if any existing NP node has two or more sub nodes
            # Check that there is a subnode of type "NN" other than the extreme left
            # Check that the descriptor is not a stop word (e.g. "the")
            if root.label() == "NP" and self.__exists_np_node(root)[0] and len(root) >= 2:
                temp_nodes = list()
                temp_list = list()

                for node in root[:self.__exists_np_node(root)[1]]:
                    temp_nodes.append(node.leaves())

                for item in temp_nodes:
                    # Descriptors must have at least 3 characters and must not be a stop word
                    for descriptor in item:
                        if len(descriptor) > 2 and not bool(set([descriptor.lower()]) & set(stopwords.words("english"))):
                            temp_list.append(descriptor)

                if len(temp_list) > 0:
                    self.descriptor_list.append(" ".join(temp_list))

            for child in root:
                self.__traverse(child)

    def get_descriptor_count(self, headline: str) -> Union[int, list]:
        """
        Retrieves descriptor count for given headline, as well as the descriptor names.

        :param headline: Headline of the newspaper article.
        :type headline: str
        :param port: Port the CoreNLP server is started on.
        :type port: int
        :raises ApiNotStarted: In case the user forgot to start the CoreNLP server.
        :return: Returns the count of descriptors as an integer, as well as the
                 list of descriptors.
        :rtype: Union[int, list]
        """
        # Filter headline for apostraphes
        headline = headline.replace("'", "")
        headline = headline.replace("’", "")
        headline = headline.replace("‘", "")

        # Connect to the Java CoreNLP Server
        parser = CoreNLPParser(self._corelnp_url)
        root = list(parser.raw_parse(headline))[0][0]
        self.__traverse(root)
        return len(self.descriptor_list), self.descriptor_list

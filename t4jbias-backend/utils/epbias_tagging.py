import json
import spacy
import os
from nltk.stem.snowball import SnowballStemmer

class EpBiasTagging:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stemmer = SnowballStemmer("english")
        self.lexicon_list_bias_tagger = '../bias_tagger_detection/lexicons'

    def _show_lemmas(self, text):
        for token in text:
            return token.lemma_

    def find_lex_epbias(self, tagged_word=None, exp_data_path=f'{os.getcwd()}/utils/explanability_data.json'):
        '''
            look for words in lexicons and capture epistemological bias type from file type.
            we use this to help us provide some explanability to the user.
        '''
        with open(exp_data_path) as jsonfile:
            jfile = json.load(jsonfile)
            epbias_types = []
            epbias_defs = []
            epbias_links = []
            #flag true when word is found in a lexicon listed in the directory
            in_lex = False
            if tagged_word != None:
                for key in jfile.keys():
                    stemmed_version = self.stemmer.stem(tagged_word)
                    lemmatised_version = self._show_lemmas(self.nlp(tagged_word))
                    if tagged_word in jfile[key]['words'] or lemmatised_version in jfile[key]['words'] or stemmed_version in jfile[key]['words']:
                        in_lex = True
                        epbias_types.append(key)
                        epbias_defs.append(jfile[key]['epbias_def'])
                        epbias_links.append(jfile[key]['link'])
                if epbias_types == []:
                    epbias_types.append("regular")
                    epbias_defs.append(jfile['other']['epbias_def'])
                    epbias_links.append(jfile['other']['link'])
            return {"in_lex": in_lex, "ep_biastype": epbias_types, "definitions": epbias_defs, "links": epbias_links}



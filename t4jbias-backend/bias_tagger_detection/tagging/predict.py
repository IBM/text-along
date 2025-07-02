from nltk.corpus import stopwords
import nltk
import json, os, re
import pandas as pd
from transformers import AutoTokenizer
import torch
import bias_tagger_detection.seq2seq.model as seq2seq_model
from bias_tagger_detection.shared.data import get_dataloader
from bias_tagger_detection.shared.args import ARGS
from bias_tagger_detection.shared.constants import CUDA
import bias_tagger_detection.tagging.model as tagging_model
import bias_tagger_detection.tagging.utils as tagging_utils
import bias_tagger_detection.seq2seq.utils as debias_utils
from tmi.tmi_logic import TMIAnalysis
import utils.utilities as gen_utils
import utils.epbias_tagging as epbias_tagging

import logging
logging.basicConfig(
    filename='app.log',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO
)

from transformers import pipeline
from transformers import AutoModelForMaskedLM

"""
python tagging/predict.py --tagger_checkpoint train_tagging/tagger_model_1.ckpt --working_dir TEST --categories_file ./bias_data/WNC/revision_topics.csv --extra_features_top --pre_enrich --activation_hidden --category_input
"""
nltk.download('stopwords', download_dir='./data/nltk_data')
import spacy
nlp = spacy.load('en_core_web_sm')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### Get stop words ###################
def get_stops():
    stops = set(stopwords.words('english'))
    return stops

################## Arguments needed for inference ######################
ARGS.extra_features_top = True
ARGS.num_categories = 43
ARGS.num_tok_labels = 3
ARGS.pre_enrich = True
ARGS.categories_file = '/data/revision_topics.csv'
ARGS.activation_hidden = True
ARGS.category_input = True
ARGS.pre_enrich = True
ARGS.bert_full_embeddings = True
ARGS.debias_weight = 1.3 
ARGS.token_softmax = True
ARGS.concat_categories = True
ARGS.category_emb = True
ARGS.add_category_emb = True
ARGS.pointer_generator = True
ARGS.bert_encoder = True

# ARGS.tagger_model_path = os.getcwd() + ARGS.tagger_model_path
if ARGS.tagger_checkpoint != None:
    ARGS.tagger_model_path = ARGS.tagger_checkpoint
else:
    # ARGS.tagger_model_path = os.getcwd() + ARGS.tagger_model_path
    ARGS.tagger_model_path = ARGS.tagger_model_path

if ARGS.debias_checkpoint != None:
    ARGS.debias_model_path = ARGS.debias_checkpoint
else:
    # ARGS.debias_model_path = os.getcwd() + ARGS.debias_model_path
    ARGS.debias_model_path = ARGS.debias_model_path

# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = AutoTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)
id2tok = {x: tok for (tok, x) in tok2id.items()}

# masked language model
autoMLM = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", cache_dir=ARGS.working_dir + '/cache')
unmasker = pipeline('fill-mask', model=autoMLM, tokenizer=tokenizer)

def add_tmi_to_data(s,  corenlp_url='https://corenlp.run/'):
    cols=['headline']
    in_data = [s]
    in_df = pd.DataFrame(in_data, columns=cols)
    tmi_obj = TMIAnalysis(df=in_df, corenlp_url=corenlp_url)
    data = tmi_obj.is_biased_from_descriptor()
    print("Done adding descriptor class\n", data['descriptor_count'], "\n")
    logging.info(f"Done adding descriptor class\n", {data['descriptor_count']}, "\n")
    return data['descriptor_count'].astype(str), data['descriptor_name'].astype(str), data['descriptor_sentiment'].astype(str), data['descriptor_class'].astype(str)
 
def words_from_toks(toks):
    words = []
    word_indices = []
    for i, tok in enumerate(toks):
        if tok.startswith('##'):
            words[-1] += tok.replace('##', '')
            word_indices[-1].append(i)
        else:
            words.append(tok)
            word_indices.append([i])
    return words, word_indices

def get_pos_dep(toks):
    out_pos, out_dep = [], []
    words, word_indices = words_from_toks(toks)
    analysis = nlp(' '.join(words))
    
    if len(analysis) != len(words):
        return None, None

    for analysis_tok, idx in zip(analysis, word_indices):
        out_pos += [analysis_tok.pos_] * len(idx)
        out_dep += [analysis_tok.dep_] * len(idx)
    
    assert len(out_pos) == len(out_dep) == len(toks)
    
    return ' '.join(out_pos), ' '.join(out_dep)

def tokenize(s, corenlpurl=None):
    logging.info("Tokenizing sentence...\n")
    logging.info(s)
    logging.info(f"Corenlp server running is: {corenlpurl}")
    try:
        pre_toks = tokenizer(
            s, 
            add_special_tokens=True, 
            max_length=512,
            padding=True,
            return_tensors='pt')['input_ids']
        for i, j in enumerate(pre_toks):
            mode = 'w' if i == 0 else 'a'
            pre_toks_from_ids = tokenizer.convert_ids_to_tokens(j)
            pre_toks_from_ids = list(filter(lambda x: x != '[PAD]', pre_toks_from_ids))
            pos_string, dep_string = '',''
            if corenlpurl != None:
                pos_string, dep_string = get_pos_dep(pre_toks_from_ids[1: len(pre_toks_from_ids)-1])
                a,b,c,d = add_tmi_to_data(s[i], corenlp_url=corenlpurl)
                print(".............", a,b,c,d)
            with open('tmp', mode) as f:
                if pos_string=='' or dep_string=='':
                    f.write('\t'.join([
                        'na',
                        ' '.join(pre_toks_from_ids[1: len(pre_toks_from_ids)-1]),
                        ' '.join(pre_toks_from_ids[1: len(pre_toks_from_ids)-1]),
                        'na',
                        'na',
                        '\n'
                    ]))
                else:
                    f.write('\t'.join([
                        'na',
                        ' '.join(pre_toks_from_ids[1: len(pre_toks_from_ids)-1]),
                        ' '.join(pre_toks_from_ids[1: len(pre_toks_from_ids)-1]),
                        'na',
                        'na',
                        pos_string,
                        dep_string,
                        a[0], # 0 because of len(s)=1
                        b[0],
                        c[0],
                        d[0],
                        '\n'
                    ]))
        logging.info("Tokenizing Done")
    except Exception as e:
        print("Error : Cannot tokenize sentence/s")
        logging.info(f"Error : Cannot tokenize sentence/s: {e}")



# # # # # # # # ## # # # ## # # MODELS # # # # # # # # ## # # # ## # #
def get_tagger_model_structure():
    logging.info("Getting tagger model structure")
    if ARGS.extra_features_top:
        tagger_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
    else:
        tagger_model = tagging_model.BertForMultitask.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache')

    return tagger_model.to(device)

def get_debias_model_structure():
    logging.info("Getting model structure")
    # # # # # # # # ## # # # ## # # MODELS # # # # # # # # ## # # # ## # #
    if ARGS.pointer_generator:
        model = seq2seq_model.PointerSeq2Seq(
            vocab_size=len(tok2id), 
            hidden_size=ARGS.hidden_size,
            emb_dim=768, # 768 = bert hidden size
            dropout=0.2, 
            tok2id=tok2id) 
    else:
        model = seq2seq_model.Seq2Seq(
            vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
            emb_dim=768, 
            dropout=0.2, 
            tok2id=tok2id)
    
    return model.to(device)

def load_tagger_model():
    global tagging_model
    tagging_model = get_tagger_model_structure()
    if os.path.exists(ARGS.tagger_model_path):
        print('LOADING TAGGER FROM ' + ARGS.tagger_model_path)
        logging.info(f'LOADING TAGGER FROM {ARGS.tagger_model_path}')
        if CUDA:
            tagging_model.load_state_dict(torch.load(ARGS.tagger_model_path))
            tagging_model = tagging_model.cuda()
        else:
            tagging_model.load_state_dict(torch.load(ARGS.tagger_model_path, map_location='cpu'))
        print('DONE.')
        logging.info("DONE loading model from file")
    else:
        msg = f"No model files found. Path checked:{ARGS.tagger_model_path}"
        logging.error(msg)
        raise FileExistsError(msg)
    tagging_model.eval()  # ensure consistent inference results
    return tagging_model

def load_debias_model():
    global debias_model
    debias_model = get_debias_model_structure()
    if os.path.exists(ARGS.debias_model_path):
        print('LOADING DEBIASER MODEL FROM ' + ARGS.debias_model_path)
        logging.info(f'LOADING DEBIASER MODEL FROM {ARGS.debias_model_path}')
        debias_model.load_state_dict(torch.load(ARGS.debias_model_path, map_location=device))
        print('DONE.')
        logging.info("DONE loading model from file")
    else:
        msg = f"No model files found. Path checked:{ARGS.debias_model_path}"
        logging.error(msg)
        raise FileExistsError(msg)
    debias_model.eval()  # ensure consistent inference results
    return debias_model

def find_special_char(word: str, char: str):
    find_idx = word.find(char)
    if find_idx != -1 and find_idx == 0:
        return True
    else:
        return False

# for cleaning and removing '##' from the results
def clean_words(words: list, tokens: list):
    special_pads = "##"
    top_cleaned = []
    for _, word in enumerate(words):
        if special_pads not in word:
            top_cleaned.append(word)
        else:
            word_before = tokens.index(word)-1
            word_after = tokens.index(word)+1
            if find_special_char(word, special_pads):
                if find_special_char(tokens[word_before], special_pads):
                    new_word = tokens[word_before-1].replace(special_pads, '') + tokens[word_before].replace(
                        special_pads, '') + word.replace(special_pads, '')
                else:
                    if find_special_char(tokens[word_after], special_pads):
                        new_word = tokens[word_before].replace(special_pads, '') + word.replace(special_pads, '') + tokens[word_after].replace(special_pads, '')
                    else:
                        new_word = tokens[word_before].replace(special_pads, '') + word.replace(special_pads, '')
                top_cleaned.append(new_word)
            else:
                if find_special_char(tokens[word_after], special_pads):
                    new_word = word + tokens[word_after].replace(special_pads, '')
                    top_cleaned.append(new_word)
                    # return top_cleaned
                top_cleaned.append(word)
    return top_cleaned


def clean_sentence(sentence: str):
    REF_RE = '(@\'!#?/\)(^%$&*-_=+\":\‘)[‘’“”…:]'
    x = sentence.lower().lstrip('@\'!#?/\)(^%$&*-_=+\":\‘')
    return re.sub('[‘’“”…:¬]', '', x)

# using a mask filler to get alternatives for the problem word
def get_alternatives(sentence: str, word_to_mask: str):
    # print(sentence, word_to_mask)
    sentence_with_masked_word = sentence.replace(word_to_mask[0], '[MASK]')
    alternatives = unmasker(sentence_with_masked_word)
    return alternatives

epbias_tagger = epbias_tagging.EpBiasTagging()
# # # # # # # # # # # # PREDICTIONS # # # # # # # # # # # # # #

def predict(s, tagger_model, debias_model=None, corenlpurl=None):
    # to get the tagged outputs for given sentences
    def tagged_outputs(s, tagger_model, debias_model=None):
        for x, y in enumerate(s):
            s[x] = str(y)
            s[x] = clean_sentence(s[x])
        if s[-1] == "":
            tokenize(s[:-1], corenlpurl)
        else:
            tokenize(s, corenlpurl)
        dl, _ = get_dataloader('tmp', tok2id, 1)
        loss_fn = tagging_utils.build_loss_fn()
        if tagger_model is None:
            print("--------------MODEL is NONE-----------")
            logging.error('Tagging Model type is None')
            raise Exception('Model type is None')
        results = tagging_utils.run_inference(tagger_model, dl, loss_fn, tokenizer)
        msg = f"\n --------------- RESULTS ------------- \n {json.dumps(results, default=gen_utils.np_encoder, indent=4)} \n"
        logging.info(msg)

        d_alternatives = []
        if debias_model != None:
            debias_hits, debias_preds, debias_golds, debias_srcs = debias_utils.run_eval(debias_model, dl, tok2id, out_file_path=f"{ARGS.working_dir}/results.txt", max_seq_len=200)
            msg = f"\n --------------- DEBIASED RESULTS ------------- \n {json.dumps(debias_preds, default=gen_utils.np_encoder, indent=4)} \n"
            logging.info(msg)
            for i in debias_preds:
                debiased_alternative, debiased_alternative_indices = words_from_toks(i)
                debiased_alternative = ' '.join(debiased_alternative[1:-1]) + "."
                d_alternatives.append(debiased_alternative)
            logging.info(d_alternatives)

        out = {}
        for i, j in enumerate(results['results']):
            top_words = [id2tok[x] for x in j['top_tok_ids']]
            input_tokens = results['input_toks'][i]
            clean_word = clean_words(top_words, input_tokens)
            if os.getenv('ExpDataPath') != None:
                exp_path = os.getenv('ExpDataPath')
                epbias_info = epbias_tagger.find_lex_epbias(clean_word[0], exp_data_path=exp_path)
            else:
                epbias_info = epbias_tagger.find_lex_epbias(clean_word[0])
            
            if debias_model != None:
                output = {'words': clean_word, 'probability': j['probability'], 'epbias_type': epbias_info['ep_biastype'],
                'epbias_def': epbias_info['definitions'], 'epbias_link': epbias_info['links'], 'in_lex': epbias_info['in_lex'], 'sentence':s[i], 'alternative':d_alternatives[i]}
            else:
                output = {'words': clean_word, 'probability': j['probability'], 'epbias_type': epbias_info['ep_biastype'],
                    'epbias_def': epbias_info['definitions'], 'epbias_link': epbias_info['links'], 'in_lex': epbias_info['in_lex'], 'sentence':s[i]}
                
            out[i] = output
        return out
    
    # to get alternatives that are less subjective.
    def alternatives():
        outputs = tagged_outputs(s, tagger_model)
        print(outputs)
        for i in outputs.keys():
            alternatives = get_alternatives(outputs[i]['sentence'], outputs[i]['words'])
            alternates = []
            for alternate in alternatives:
                alternates.append(alternate['sequence'])
            # print("ALTERNATESSS ", alternates)
            new_predictions = tagged_outputs(s=alternates, tagger_model=tagger_model)
            # print("new predictions ", new_predictions)
            outputs[i]['alternative'] = ""
            subjective_list = []
            subjective_probs = []
            for j in new_predictions.keys():
                if 'subjectives' in new_predictions[j]['epbias_type']:
                    subjective_list.append(new_predictions[j]['sentence'])
                    subjective_probs.append(new_predictions[j]['probability'])
                    continue
                else:
                    outputs[i]['alternative'] = new_predictions[j]['sentence']
            new_sentence = []
            if outputs[i]['alternative'] == "":
                outputs[i]['alternative'] = (outputs[i]['sentence']).replace(outputs[i]['words'][0], "")
                new_sentence.append(subjective_list[subjective_probs.index(min(subjective_probs))])
                return predict(new_sentence, tagger_model)
        out = json.dumps(outputs, default=gen_utils.np_encoder)
        logging.info(out)
        return out
    
    if debias_model != None:
        print("------- Not NONE ------")
        return tagged_outputs(s, tagger_model, debias_model)
    else:
        print("Debias model is none, so using self analysing alternatives which takes longer.")
        return alternatives()


# ####### only use for testing via input entry #####
# # get_pos_rel("What is the airspeed of an unladen swallow ?")
# taggermodel = load_tagger_model()
# debiasmodel = load_debias_model()
# # ### Use for single sentence example test
# # s = ["Two white guys walked down the street and needlessly murdered a black guy.", "Zuckerberg claims Facebook can revolutionise the world.", "Controlling husband of Eddy Grant's niece is found guilty of murdering her."]
# s1 = ["Zuckerberg claims Facebook can revolutionise the world.", "Controlling husband of Eddy Grant's niece is found guilty of murdering her."]
# # s2 = ["Meghan gets political again, Duchess weighs in on Biden's pick of Ketanji Brown Jackson for the Supreme Court and says she opens new ground for representation."]
# print(predict(s1,taggermodel, debiasmodel))

# while True:
#     s = input("Enter a sentence: ")
#     # start = time.time()
#     print(predict(s,taggermodel))
    # print("Took %.2f seconds" % (time.time() - start))
# quit()
# ##################################################

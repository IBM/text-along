import torch
from transformers import AutoTokenizer
import pandas as pd
import json, os, re
import bias_tagger_detection.seq2seq.model as seq2seq_model
from bias_tagger_detection.shared.args import ARGS
from bias_tagger_detection.shared.data import get_dataloader
import bias_tagger_detection.seq2seq.utils as debias_utils
from .tmi.tmi_logic import TMIAnalysis
import .utils.utilities as gen_utils

import logging
logging.basicConfig(
    filename=f'{__file__}.log',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO
)

import spacy
nlp = spacy.load('en_core_web_sm')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARGS.working_dir = os.path.abspath(os.path.dirname(__file__))
ARGS.pointer_generator = True
ARGS.bert_full_embeddings = True
ARGS.bert_encoder = True
print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
ARGS.debias_checkpoint = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/seq2seq-output-2/debiaser_20.ckpt"
if ARGS.debias_checkpoint == None:
    raise "No Debiasing Models Provided."


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = AutoTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)
id2tok = {x: tok for (tok, x) in tok2id.items()}



def add_tmi_to_data(s,  corenlp_url='https://corenlp.run/'):
    cols=['headline']
    in_data = [s]
    in_df = pd.DataFrame(in_data, columns=cols)
    tmi_obj = TMIAnalysis(df=in_df, corenlp_url=corenlp_url)
    data = tmi_obj.is_biased_from_descriptor()
    return data['descriptor_count'].astype(str), data['descriptor_name'].astype(str), data['descriptor_sentiment'].astype(str), data['descriptor_class'].astype(str)
 
def get_pos_dep(toks):
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

def clean_sentence(sentence: str):
    REF_RE = '(@\'!#?/\)(^%$&*-_=+\":\‘)[‘’“”…:]'
    x = sentence.lower().lstrip('@\'!#?/\)(^%$&*-_=+\":\‘')
    return re.sub('[‘’“”…:¬]', '', x)

def tokenize(s):
    logging.info("Tokenizing sentence...\n")
    logging.info(s)
    try:
        pre_toks = tokenizer(
            s, 
            add_special_tokens=True, 
            max_length=512,
            padding=True,
            return_tensors='pt')['input_ids']
    except TypeError:
        print("Error : Cannot tokenize sentence/s")
    for i, j in enumerate(pre_toks):
        mode = 'w' if i == 0 else 'a'
        pre_toks_from_ids = tokenizer.convert_ids_to_tokens(j)
        pre_toks_from_ids = list(filter(lambda x: x != '[PAD]', pre_toks_from_ids))
        pos_string, dep_string = '',''
        pos_string, dep_string = get_pos_dep(pre_toks_from_ids[1: len(pre_toks_from_ids)-1])
        a,b,c,d = add_tmi_to_data(s[i], corenlp_url='https://corenlp.run/')
        with open('tmp', mode) as f:
            if pos_string is None or dep_string is None:
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

# # # # # # # # ## # # # ## # # MODEL # # # # # # # # ## # # # ## # #
def get_model_structure():
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

def load_debias_model():
    global debias_model
    debias_model = get_model_structure()
    if os.path.exists(ARGS.debias_checkpoint):
        print('LOADING MODEL FROM ' + ARGS.debias_checkpoint)
        logging.info(f'LOADING MODEL FROM {ARGS.debias_checkpoint}')
        debias_model.load_state_dict(torch.load(ARGS.debias_checkpoint, map_location=device))
        print('DONE.')
        logging.info("DONE loading model from file")
    else:
        msg = f"No model files found. Path checked:{ARGS.debias_checkpoint}"
        logging.error(msg)
        raise FileExistsError(msg)
    debias_model.eval()  # ensure consistent inference results
    return debias_model
    
def debias_output(s, model):
    for x, y in enumerate(s):
        s[x] = str(y)
        s[x] = clean_sentence(s[x])
    if s[-1] == "":
        tokenize(s[:-1])
    else:
        tokenize(s)
    dl, _ = get_dataloader('tmp', tok2id, 1)
    hits, preds, golds, srcs = debias_utils.run_eval(model, dl, tok2id, out_file_path=f"{ARGS.working_dir}/results.txt", max_seq_len=200)
    msg = f"\n --------------- RESULTS ------------- \n {json.dumps(preds, default=gen_utils.np_encoder, indent=4)} \n"
    print(msg)
    logging.info(msg)
    
###### only use for testing via input entry #####
model = load_debias_model()
### Use for single sentence example test
s = ["Two white guys walked down the street and needlessly murdered a black guy.", "Zuckerberg claims Facebook can revolutionise the world.", "Controlling husband of Eddy Grant's niece is found guilty of murdering her."]
# s = ["Zuckerberg claims Facebook can revolutionise the world.", "Controlling husband of Eddy Grant's niece is found guilty of murdering her."]
# s2 = ["Meghan gets political again, Duchess weighs in on Biden's pick of Ketanji Brown Jackson for the Supreme Court and says she opens new ground for representation."]
print(debias_output(s,model))

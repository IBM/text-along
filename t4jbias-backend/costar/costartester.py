import json, os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bias_tagger_detection.shared.args import ARGS
import utils.utilities as utils

import logging
logging.basicConfig(
    filename='app.log',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO
)

ARGS.costar_model_path = ARGS.costar_model_path
SC_MODEL_PATH = ARGS.costar_model_path + '/CO-STAR-sc'
CS_MODEL_PATH = ARGS.costar_model_path + '/CO-STAR-cs'
SBF_MODEL_PATH = ARGS.costar_model_path + '/SBF-GPT2'
GPT_MODEL = 'gpt2'
BADWORDS_PATH = ARGS.costar_model_path + '/badwords.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############ Tokenize #####################
tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL, cache_dir=ARGS.working_dir + '/costar/cache')
tokenizer.add_special_tokens({'sep_token': '[SEP]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
#########################################

################# Get Bad Words #################
def getbadwords():
    f = open(BADWORDS_PATH)
    data = json.load(f)
    badwords_input_ids = tokenizer(data, add_prefix_space=True).input_ids
    f.close()

    return badwords_input_ids
#############################################

############# General Arguments ####################
# settings for beam search decoding; change for top-k
GEN_ARGS = {
    'repetition_penalty': 2.5,
    'length_penalty': 2.5,
    'early_stopping': True,
    'use_cache': True,
    'num_return_sequences': 3,
    'no_repeat_ngram_size': 2,
    'num_beams': 3,
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'bad_words_ids': getbadwords()
}
#########################################


################# Clean Text #################
def cleantext(cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_outputs):
    cs_stereotypes = [s.lower().strip() for s in cs_stereotypes]
    cs_stereotypes = [s.rstrip('.') for s in cs_stereotypes]
    sc_stereotypes = [s.lower().strip() for s in sc_stereotypes]
    sc_stereotypes = [s.rstrip('.') for s in sc_stereotypes]

    cs_concepts = [c.lower().strip() for c in cs_concepts]
    cs_concepts = [c.rstrip('.') for c in cs_concepts]
    sc_concepts = [c.lower().strip() for c in sc_concepts]
    sc_concepts = [c.rstrip('.') for c in sc_concepts]

    sbf_stereotypes = [s.lower().strip() for s in sbf_outputs]
    sbf_stereotypes = [s.rstrip('.') for s in sbf_stereotypes]
    
    return cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes
#############################################

################# Get Phrase to Test ############################
def getentry(s):
    logging.info(f"COSTAR getentry {s}")
    for index, sentence in enumerate(s):
        sentence = sentence.lower().strip()
        sentence += ' [SEP] '
        s[index] = sentence
    return s
#############################################

##################### Print Results ########################
def output_results(cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes):
    print('==================================================')
    print(' DEMO FOR MODELS TRAINED ON THE CO-STAR FRAMEWORK')
    print('==================================================')

    print('')
    print('-------------------  CO-STAR  --------------------')
    print('CS Stereotypes:')
    print(*cs_stereotypes, sep='\n')
    print('SC Stereotypes:')
    print(*sc_stereotypes, sep='\n')
    print('')
    print('CS Concepts:')
    print(*cs_concepts, sep='\n')
    print('SC Concepts:')
    print(*sc_concepts, sep='\n')
    print('')
    print('--------------  Social Bias Frames  --------------')
    print('Stereotypes:')
    print(*sbf_stereotypes, sep='\n')
    print('')
    return [*cs_stereotypes, *sc_stereotypes, *cs_concepts, *sc_concepts, *sbf_stereotypes]
#############################################

################# Run Model #################
def run_model(post,cs_model, sc_model, sbf_model):
    logging.info("COSTAR run_model")
    encoded_post = tokenizer(post, padding=True, truncation=True)
    input_ids = torch.tensor(encoded_post['input_ids'])
    attention_mask = torch.tensor(encoded_post['attention_mask'])

    # max sentence length of the post is the length of any sentence in post - each tokenized sentence is padded to the same length
    max_len = len(input_ids[0].squeeze().numpy()) + 50
    cs_model.config.max_length = sc_model.config.max_length = sbf_model.config.max_length = max_len

    cs_outputs = cs_model.generate(input_ids=input_ids, attention_mask=attention_mask, **GEN_ARGS)
    sc_outputs = sc_model.generate(input_ids=input_ids, attention_mask=attention_mask, **GEN_ARGS)
    sbf_outputs = sbf_model.generate(input_ids=input_ids, attention_mask=attention_mask, **GEN_ARGS)

    cs_outputs = tokenizer.batch_decode(cs_outputs)
    sc_outputs = tokenizer.batch_decode(sc_outputs)
    sbf_outputs = tokenizer.batch_decode(sbf_outputs)
    
    logging.info(f"Decoded raw outputs:\n{cs_outputs},\n{sc_outputs},\n{sbf_outputs}")

    # loop through all of the sentences in the post; post is a list
    cs_outputs_final, sc_outputs_final, sbf_outputs_final = [],[],[]

    begin_index = 0
    for sentence in post:
        # replace the sentence itself and eos_tokens because eos_token exist on both left and right sides of a sentence due to padding left
        # not set padding_side = left produces weird decoded results (I quite disagree with the results without padding left)
        cs_outputs_final += [cs_outputs[i].replace(sentence, "").replace(tokenizer.eos_token, "") for i in range(begin_index, begin_index + 3)]
        sc_outputs_final += [sc_outputs[i].replace(sentence, "").replace(tokenizer.eos_token, "") for i in range(begin_index, begin_index + 3)]
        sbf_outputs_final += [sbf_outputs[i].replace(sentence, "").replace(tokenizer.eos_token, "") for i in range(begin_index, begin_index + 3)]
        begin_index += 3 

    cs_outputs = [output.split('[SEP]') for output in cs_outputs_final]
    cs_outputs = [output for output in cs_outputs if len(output) == 2]
    sc_outputs = [output.split('[SEP]') for output in sc_outputs_final]
    sc_outputs = [output for output in sc_outputs if len(output) == 2]

    cs_stereotypes = [output[1] for output in cs_outputs]
    cs_concepts = [output[0] for output in cs_outputs]
    sc_stereotypes = [output[0] for output in sc_outputs]
    sc_concepts = [output[1] for output in sc_outputs]
    logging.info(f"{cs_stereotypes},{cs_concepts},{sc_stereotypes}, {sc_concepts}, {sbf_outputs_final}")
    logging.info("COSTAR run_model finished")
    # print('TESTing outs ', cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_outputs_final)
    return cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_outputs_final
#############################################

################# Load Models #################
def load_costar_model():

    if os.path.exists(ARGS.costar_model_path):
        print('LOADING COSTAR FROM ' + ARGS.costar_model_path)
        logging.info(f'LOADING COSTAR FROM {ARGS.costar_model_path}')

        sc_model = GPT2LMHeadModel.from_pretrained(SC_MODEL_PATH, cache_dir=ARGS.working_dir + '/costar/cache').to(device)
        cs_model = GPT2LMHeadModel.from_pretrained(CS_MODEL_PATH, cache_dir=ARGS.working_dir + '/costar/cache').to(device)
        sbf_model = GPT2LMHeadModel.from_pretrained(SBF_MODEL_PATH, cache_dir=ARGS.working_dir + '/costar/cache').to(device)

        cs_model.eval()
        sc_model.eval()
        sbf_model.eval()
        # cs_costar_model, sc_costar_model,sbf_model = costar_run.load_costar_model()
        print('DONE.')
        logging.info("DONE loading model from file")
    else:
        msg = f"No model files found. Path checked:{ARGS.costar_model_path}"
        logging.error(msg)
        raise FileExistsError(msg)

    return cs_model, sc_model, sbf_model
#############################################


def run_inference(s, cs_model, sc_model, sbf_model):
    logging.info("COSTAR run_inference")
    with torch.no_grad():
        #while True:    
            post =  getentry(s)
            cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_outputs = run_model(post,cs_model, sc_model, sbf_model) #tensor encoding,  run model on encoding, and get output
            cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes = cleantext(cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_outputs) #clean output and format output into a list
            logging.info(f"{cs_stereotypes},{cs_concepts},{sc_stereotypes},{sc_concepts},{sbf_stereotypes}")
            return cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes

def predict(s, cs_costar_model, sc_costar_model,sbf_model):
    '''
    Uses the COSTAR and Social Bias Frame (SBF) models 
    to detect stereotypes and their associated concepts.
    Then a semantic similarity is computed between the sentence
    and a corpus of the stereotypes and concepts. 
    Which helps with ranking the stereotypes and concepts.

    '''
    cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes = run_inference(s, cs_costar_model, sc_costar_model,sbf_model)
    results = output_results(cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes)
    msg = f"\n --------------- RESULTS -------------\n{json.dumps(results, default=utils.np_encoder, indent=4)}\n"
    logging.info(msg)

    #loop through each sentence and match results from models to each sentence
    sentence_output_pair = {}
    begin_i = 0
    for index, sentence in enumerate(s):
        # match each input sentence to their 3 corresponding stereotype and concept
        stereotype_corpus= list(set(cs_stereotypes[begin_i:begin_i+3] + sc_stereotypes[begin_i:begin_i+3] + sbf_stereotypes[begin_i:begin_i+3])) 
        concept_corpus= list(set(cs_concepts[begin_i:begin_i+3] + sc_concepts[begin_i:begin_i+3]))
        begin_i += 3

        ranked_stereotypes = utils.semantic_search(sentence,stereotype_corpus)
        ranked_concepts = utils.semantic_search(sentence,concept_corpus)
        msg = f"\n ----- SORTED STEREOTYPE LIST ---- \n {ranked_stereotypes}"
        logging.info(msg)
        msg2 = f"\n ----- SORTED CONCEPT LIST ---- \n {ranked_concepts}"
        logging.info(msg2)
        
        # ranked_stereotypes['sentences'] -> [[]] with only one item
        out = {"stereotypes":ranked_stereotypes['sentences'][0],"stereotype_distances":ranked_stereotypes['distance_scores'][0],
        "concepts": ranked_concepts['sentences'][0], "concept_distances": ranked_concepts['distance_scores'][0]}
        logging.info(out)
        sentence_output_pair[index] = out
    return json.dumps(sentence_output_pair, default=utils.np_encoder)

# ################# To Run independently####################### 
# post = ['Zuckerberg claims Facebook can revolutionise the world.']
# print('Loading Model')
# cs_model, sc_model, sbf_model = load_costar_model() #if this is uncommented be sure the change the run_inference parameters
# print('Models Loaded')
# cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes = run_inference(post, cs_model, sc_model, sbf_model)
# # cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes = run_model(post,cs_model, sc_model, sbf_model) #tensor encoding,  run model on encoding, and get output
# output_results(cs_stereotypes,cs_concepts,sc_stereotypes, sc_concepts, sbf_stereotypes)
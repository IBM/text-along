from re import T
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from bias_tagger_detection.shared.args import ARGS
import bias_tagger_detection.tagging.predict as tagger_run
import costar.costartester as costar_run
import utils.utilities as utils
import urllib.request
import requests
from bs4 import BeautifulSoup

import logging
logging.basicConfig(
    filename='app.log',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO
)
logging.info("Finished all imports successfully")
logging.info('Started..........')

"""
Contains the main functions for serving up the analysis server to detect subjective bias in text. 
Can be run with ```python app.py```
"""
#NOTE: you need to download models using the helper_scripts/download-models.sh script.

ARGS.working_dir = 'APP'

def getpage(url):
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    html = mybytes.decode("unicode_escape")
    fp.close()
    return html

def get_main_content(html_doc):
    text = BeautifulSoup(html_doc, 'html.parser')
    src_text = text.find(class_="scrtext")
    src_text = src_text.find('pre')
    return str(src_text)

def analyse_url(url):
    if url is None or url == '':
        return "error: bad request", 400
    try:
        html = getpage(url)
        main_html = get_main_content(html)
    except:
        return "The 'url' provided could not be loaded.", 400

def get_headline_from_url(url):
    text = []
    if url is None or url == '':
        msg = "error: bad request, url is none or empty"
        logging.error(msg)
        raise Exception(msg)
    else:
        try:
            get_url_text = requests.get(url).text
            soup = BeautifulSoup(get_url_text, "html.parser")
            get_headers = soup.select("h1")
            # print(get_headers[0].text.strip())
            text.append(get_headers[0].text.strip()+".\n")
            get_body = [p.get_text().strip()+"\n" for p in soup.find_all("p")]
            text = text + get_body
            # print(text)
            return {'text': text}
        except Exception as e:
            logging.error(e)
            return e, 400

class Text(BaseModel):
    text: list

class URL(BaseModel):
    url: str
#################################### API #######################################
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["tagger"] = tagger_run.load_tagger_model()
    ml_models["debiaser"] = tagger_run.load_debias_model()
    ml_models['cs_model'], ml_models['sc_model'], ml_models['sbf_model'] = costar_run.load_costar_model()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get('/')
async def root():
    return {'msg': 'Try POSTing to the /predict/tagger or /predict/costar endpoints with a url and headline text'}

@app.post('/predict/tagger')
async def tagger_route(text: Text):
    try:
        req_data = text.text
        try:
            print("\n ---------TEXT-------- ", req_data)
            logging.info(f"---------TEXT-------- {req_data}")
            prediction = tagger_run.predict(req_data, ml_models["tagger"], ml_models["debiaser"], corenlpurl=ARGS.corenlpurl)
            logging.info(prediction)
            return json.dumps(prediction, default=utils.np_encoder)
        except Exception as e:
            logging.error(e)
            return f"error: {e}", 400
    except HTTPException as e:
        logging.error(f"error: The key 'text' is missing.{e}")
        raise HTTPException(status_code=400, detail=f"error: The key 'text' is missing.{e}")
   
@app.post('/predict/costar')
async def costar_route(text: Text):
    try:
        req_data = text.text
        try:
            print("\n ---------TEXT-------- ", req_data)
            logging.info(f"---------TEXT-------- {req_data}")
            prediction = costar_run.predict(req_data, ml_models['cs_model'], ml_models['sc_model'], ml_models['sbf_model'])
            logging.info(prediction)
            return prediction
        except Exception as e:
            logging.error(e)
            return f"error: {e}", 400
    except HTTPException as e:
        logging.error(f"error: The key 'text' is missing.{e}")
        raise HTTPException(status_code=400, detail=f"error: The key 'text' is missing.{e}")

@app.post('/predict/get_headline_from_url')
async def get_url_headline(body: URL):
    try:
        url = body.url
        print("\n ---------URL PATH-------- ", url)
        try:
            get_text = get_headline_from_url(url)['text']
            return json.dumps({'text':get_text}, default=utils.np_encoder)
        except Exception as e:
            logging.error(e)
            return f"error: {e}", 400
    except HTTPException as e:
        logging.error(f"error: The key 'url' is missing.{e}")
        raise HTTPException(status_code=400, detail=f"error: The key 'url' is missing.{e}")


if __name__ == '__main__':
    try:
        uvicorn.run("app:app", host='0.0.0.0', port=6006, log_level=1, reload=True)
        print("\n ...............INFO: Server stopped.............")
        logging.info("...............INFO: Server stopped.............")
    except Exception as e:
        print(f"error: {e}\n")
        logging.error(e)


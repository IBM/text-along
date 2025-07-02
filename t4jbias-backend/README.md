## **Installation**

1. Clone the repo, then create and activate a python environment e.g. 
```#tested with python 3.11.4``` 
```
python3 -m venv venv 
```
2. Then 
```
source ./venv/bin/activate
```
3. Then install requirements file with `pip install -r requirements.txt` and also as a pip module `pip install -e .`
4. Run 
```
python -m spacy download en_core_web_sm
```

## **Running the server** 
Run the flask app to access the api endpoint for different detection workloads.

- Download and unzip the model files (from [here](https://huggingface.co/datasets/Lamogha/text-along-models-zip/resolve/main/models.zip?download=true)) into a `models` directory. 

#### (Optional) Configure/Start CoreNLP server
Have the CoreNLP Server downloaded and started. Your options are:

1. Start a corenlp server - the server is available for download [here](https://stanfordnlp.github.io/CoreNLP/history.html). (Tested for version 4.2.2)
To start the server enter the following command in the terminal:

```shell
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000 &
```
## **Running the App Backend Server**
Make sure you've changed directory into the back-end folder 
```
cd t4jbias-backend
```

If the optional step is skipped, simply run  
```
python app.py
```

OR if not

```
python app.py --corenlpurl "<your corenlp server url>"
```

**Note:** 
1. If this optional step is skipped, simply run ```python app.py```
2. If the above line throws an error regarding NLTK, run the following line **before** running the app. 

```
export NLTK_DATA=./data/nltk_data
```

To test that the backend is running, you can run the following command:
```
curl --location 'http://127.0.0.1:6006/predict/tagger/' --header 'Content-Type: application/json' --data '{"text": ["Zuckerberg claims Facebook can revolutionise the world."]}'
```
and

```
curl --location 'http://127.0.0.1:6006/predict/costar' --header 'Content-Type: application/json' --data '{"text": ["Zuckerberg claims Facebook can revolutionise the world."]}'
```

Congratulations! You have the backend running, now you are ready to launch the front-end.
## **Running the Frontend**

CD in the front-end folder 
```
cd t4jbias-frontend
```
Instructions for running the frontend can be found [here](https://github.com/IBM/text-along/blob/main/t4jbias-frontend/README.md).


## **Files and Folders Descriptions**

Please see [this](https://github.com/IBM/text-along/blob/main/docs/files_and_folders.md) file for descriptions of files and folders.

# Too Much Information (TMI) Preliminary Experiment
---
## Code Structure
<img src="../../images/code_structure.png">

## Prerequsites for using Code in /src/tmi/python_code
#### 1. Install required python packages
If not already done.
`pip3 install -r requirements.txt`

#### 2. Configure environment variables
In the root folder of the repository create an env file containing the following variables:
`NLU_API_KEY=${YOUR_API_KEY}`
`NLU_SERVICE_URL=${YOUR_SERVICE_URL}`

Put the Watson Natural Language Understanding API Key & Service URL into the .env file.
You can find this information on IBM Cloud, provided you have configured a Watson Natural Language Understanding instance. If you have not done so yet, please configure an instance [here](https://cloud.ibm.com/catalog/services/natural-language-understanding).

#### 3. Configure/Start CoreNLP server
Have the CoreNLP Server downloaded and started. The server is available for download [here](https://stanfordnlp.github.io/CoreNLP/history.html). (Tested for version 4.2.2)
To start the server enter the following command in the terminal:

```shell
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000 &
```
#### 4. Configure/Start NodeJS API
Have the NodeJS API available at src/tmi/readibility-api started.
If using the API for the first time, in the terminal enter:
```shell
npm install
```
Once the necessary packages are installed, enter:
```shell
npm start
```

## Using the tmi_logic.py file
The tmi_logic.py file contains the class `TMIAnalysis`. An object of the class can be created
by instantiating the class passing it a list of articles as a parameter. 
All news apis contained in /src/news_apis return a <b>standardised output</b> that can be passed 
to the class directly without further cleaning needed.


<u>Example:</u>

```python
guardian_instance = GuardianNewsAPI(GuardianAPIKey, Watson_NLU_APIKey, Watson_NLU_ServiceURL)
the_guardian_articles = guardian_instance.get_articles("murder")
my_analysis = TMIAnalysis(the_guardian_articles)
```

## Datapipeline for src/tmi
<img src="../../images/data_pipeline_tmi.png">

## Preview of analysis
A previews of all analysis performed can be viewed in the python notebook <b>("tmi.ipynb")</b> located in /src/tmi.

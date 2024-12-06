## **Files and Folders Descriptions**
### **t4jbias-backend/bias_tagger_detection**

Contains source files for training or testing the tagger model, used for subjective framing bias detection.

### t4jbias-backend/costar**
This file contains python code for utilising the costar model to generate potential stereotypes and stereotype concepts for a given sentence.

### **t4jbias-backend/tmi**

Contains source files for experiment that attempts to see if there is Too Much Information (TMI) in given news articles and add this additional attribute to analysis.

### **t4jbias-backend/utils**

This folder would contains `utilities.py` and `epbias_tagging.py` utilities file that would have general-purpose functions like sentiment analysis, that can be imported as a package. It currently has these capabilities:

- sentiment analysis
- converting an articles_text to JSON
  - articles_text is a Python dict with this schema: keys are tuples: (article title, article url), value is the text
- taking the text of an article, and counting how many times words from a given list appear in the article (e.g counting the amount of factive verbs in an article)
- utilities relating to doing the epistemological lexicon lookup.

### **data folder**

This folder contains files that can be used to conduct exploratory experiments.
For example, there is a text file that has a list of bias-inducing words.

# CAMPHOR- Advent Calendar Analysis
Tool for analyzing entries of CAMPHOR- Advent Calendar

## Set up
Install the latest Python 3 and Mecab with a dictionary.
Use of [mecab-ipadic-neologd](https://github.com/neologd/mecab-ipadic-neologd) can achieve better result.
```
python3 -m venv venv
echo $(pwd)/advent > venv/lib/python3.9/site-packages/_advent.pth
pip install -r advent/requirements.txt
```

### Run scripts
```
# Download entries in HTML
python3 analyze.py download
# Extract content as plain text
python3 analyze.py convert
# Analyze
python3 analyze.py summary
python3 analyze.py count-words
python3 analyze.py tfidf
python3 analyze.py doc2vec
```
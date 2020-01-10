import pandas as pd
import spacy
from tqdm import tqdm

df = pd.read_csv('./data/paper_title_abstract.tsv',delimiter='\t')
titles = df.PaperTitle.to_list()
abstracts = df.Abstract.to_list()
nlp = spacy.load("en_core_web_sm")
with open('./data/corpus.txt','w') as file:
    abstract = tqdm(abstracts)
    text = ''
    for abst in abstract:
        doc = nlp(abst)
        for sent in doc.sents:
            text += sent.text + '\n'
        text += '\n'
    print(text,file=file)
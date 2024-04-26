import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest




'''def get_simple_word(word):
    
    synsets = wordnet.synsets(word)
    if synsets:
        lemmas = synsets[0].lemmas()
        if lemmas:
            return lemmas[0].name()
    return word'''

def summarizer(rawdocs):
    stop_word= list(STOP_WORDS)
    nlp= spacy.load("en_core_web_sm")
    doc =nlp(rawdocs)
    
    legal_words = {"law", "legal", "court", "judge","rule","accused","jury","lawyer","victim"}  
    contains_legal_words = any(token.text.lower() in legal_words for token in doc)

    if not contains_legal_words:
        return "Error: The text is not related to the legal sector.",doc,len(rawdocs.split(' ')),None
    
    word_freq={}
    for word in doc:
        if word.text.lower() not in stop_word and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text]=1
            else:
                word_freq[word.text]+=1


    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word]=word_freq[word]/max_freq

        
    sent_tokens = [sent for sent in doc.sents]
    #print(sent_tokens)

    sent_score={}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent]= word_freq[word.text]
                else:
                    sent_score[sent] +=word_freq[word.text]

    select_len = int(len(sent_tokens)*0.4)

    summary= nlargest(select_len,sent_score,key= sent_score.get)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary,doc, len(rawdocs.split(' ')), len(summary.split(' '))





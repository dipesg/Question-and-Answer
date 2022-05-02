import numpy as np
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from similarity.normalized_levenshtein import NormalizedLevenshtein
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from utils import logger

file_object = open("./Logs/updatedhelper_log.txt", 'a+')
log_writer = logger.App_Logger()
    
def filter_same_sense_words(original, wordlist):
    filtered_words=[]
    base_sense =original.split('|')[1] 
    print (base_sense)
    for eachword in wordlist:
        if eachword[0].split('|')[1] == base_sense:
            filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
    return filtered_words
    
def get_highest_similarity_score(wordlist, wrd):
    normalized_levenshtein = NormalizedLevenshtein()
    score=[]
    for each in wordlist:
        score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
    return max(score)
    
def sense2vec_get_words(word, topn, question):
    output = []
    print ("word ",word)
    try:
        s2v = Sense2Vec().from_disk('../s2v/s2v_old')
        sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
        most_similar = s2v.most_similar(sense, n=topn)
        # print (most_similar)
        output = filter_same_sense_words(sense,most_similar)
        print ("Similar ",output)
    except:
        output =[]

        threshold = 0.6
        final=[word]
        checklist =question.split()
        for x in output:
            if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
                final.append(x)
            
        return final[1:]
    
def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)
    return [words[idx] for idx in keywords_idx]
    
def get_distractors_wordnet(word):
    distractors=[]
    try:
        syn = wn.synsets(word,'n')[0]
            
        word= word.lower()
        orig_word = word
        if len(word.split())>0:
            word = word.replace(" ","_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0: 
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            #print ("name ",name, " word",orig_word)
            if name == orig_word:
                continue
            name = name.replace("_"," ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
    except:
        print("Wordnet distractors not found")
    return distractors
    
def get_distractors (word, origsentence, top_n, lambdaval):
    distractors = sense2vec_get_words(word, top_n, origsentence)
    sentencemodel = SentenceTransformer('msmarco-distilbert-base-v3')
    print ("distractors ",distractors)
    if len(distractors) ==0:
        print("Empty solve the issue!!")
    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)
    # print ("distractors_new .. ",distractors_new)

    embedding_sentence = origsentence+ " "+word.capitalize()
    # embedding_sentence = word
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
    max_keywords = min(len(distractors_new),5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
    # filtered_keywords = filtered_keywords[1:]
    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() !=word.lower():
            final.append(wrd.capitalize())
    final = final[1:]
    return final
    
if __name__ == "__main__":
    sent = "What cryptocurrency did Musk rarely tweet about?"
    keyword = "Bitcoin"
    #final = uphelper.sense2vec_get_words(keyword, 5, sent)
    #print(final)
    #print (get_distractors(keyword, sent, 40, 0.2))
    print(sense2vec_get_words(keyword, 5, sent))
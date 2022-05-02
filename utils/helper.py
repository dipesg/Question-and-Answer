import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import random
import numpy as np
from utils import logger

class Helper:
    def __init__(self):
        # Initializing the logger object
        self.file_object = open("./Logs/helper_log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def postprocesstext (self, content):
        try:
            final=""
            for sent in sent_tokenize(content):
                sent = sent.capitalize()
                final = final +" "+sent
            return final
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the postprocesstext function. Error:: %s' % ex)
            raise ex
    
    def summarizer(self, text):
        """
        Text summarization using T5-base model.
        """
        try:
            model = T5ForConditionalGeneration.from_pretrained('t5-base')
            tokenizer = T5Tokenizer.from_pretrained('t5-base')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            text = text.strip().replace("\n"," ")
            text = "summarize: "+text
            max_len = 512
            encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

            outs = model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            early_stopping=True,
                                            num_beams=3,
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            min_length = 75,
                                            max_length=300)


            dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
            summary = dec[0]
            summary = self.postprocesstext(summary)
            summary= summary.strip()
            return summary
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the summerizer function. Error:: %s' % ex)
            raise ex
    
    def get_nouns_multipartite(self, content):
        """ 
        Answer Span Extraction (Keywords and Noun Phrases)
        """
        out=[]
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=content)
            #    not contain punctuation marks or stopwords as candidates.
            pos = {'PROPN','NOUN'}
            #pos = {'PROPN','NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.candidate_selection(pos=pos, stoplist=stoplist)
            # build the Multipartite graph and rank candidates using random walk,
            # alpha controls the weight adjustment mechanism, see TopicRank for
            # threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
            keyphrases = extractor.get_n_best(n=15)
            
            for val in keyphrases:
                 out.append(val[0])
            return out
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the summerizer function. Error:: %s' % ex)
            raise ex
    
    def get_keywords(self, originaltext,summarytext):
        try:
            keywords = self.get_nouns_multipartite(originaltext)
            print ("keywords unsummarized: ",keywords)
            keyword_processor = KeywordProcessor()
            for keyword in keywords:
                keyword_processor.add_keyword(keyword)

            keywords_found = keyword_processor.extract_keywords(summarytext)
            keywords_found = list(set(keywords_found))
            print ("keywords_found in summarized: ",keywords_found)

            important_keywords =[]
            for keyword in keywords:
                if keyword in keywords_found:
                    important_keywords.append(keyword)

            return important_keywords[:4]
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the get_keywords function. Error:: %s' % ex)
            raise ex
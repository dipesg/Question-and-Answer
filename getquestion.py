import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from textwrap3 import wrap
from utils import logger
from utils.helper import Helper
class Question:
    def __init__(self):
        self.file_object = open("./Logs/question_log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        
    def get_question(self, context, answer):
        try:
            model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
            tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            text = "context: {} answer: {}".format(context,answer)
            encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

            outs = model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            early_stopping=True,
                                            num_beams=5,
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            max_length=72)


            dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


            Question = dec[0].replace("question:","")
            Question= Question.strip()
            return Question
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the get_question function. Error:: %s' % ex)
            raise ex
        
if __name__ == "__main__":
    text = text = """When there are one or more inputs you can use a process of optimizing the values of the coefficients 
    by iteratively minimizing the error of the model on your training data.This operation is called Gradient Descent and works
    by starting with random values for each coefficient. The sum of the squared errors are calculated for each pair of input 
    and output values. A learning rate is used as a scale factor and the coefficients are updated in the direction towards minimizing
    the error. The process is repeated until a minimum sum squared error is achieved or no further improvement is possible.When using
    this method, you must select a learning rate (alpha) parameter that determines the size of the improvement step to take on each
    iteration of the procedure.Gradient descent is often taught using a linear regression model because it is relatively 
    straightforward to understand. In practice, it is useful when you have a very large dataset either in the number of rows or the
    number of columns that may not fit into memory."""
    for wrp in wrap(text, 150):
      print (wrp)
    print ("\n")
    helper = Helper()
    question = Question()
    summarized_text = helper.summarizer(text)
    imp_keywords = helper.get_keywords(text,summarized_text)
    for answer in imp_keywords:
        ques = question.get_question(summarized_text,answer)
        print (ques)
        print (answer.capitalize())
        print ("\n")
    
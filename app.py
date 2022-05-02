from utils.helper import Helper
from utils.updatedhelper import get_distractors_wordnet, get_distractors
from updatedgetquestion import Question
from textwrap3 import wrap
import gradio as gr
context = gr.inputs.Textbox(lines=10, placeholder="Enter paragraph/content here...")
output = gr.outputs.HTML(  label="Question and Answers")
radiobutton = gr.inputs.Radio(["Wordnet", "Sense2Vec"])

def generate_question(context, radiobutton):
    try:
        ques = Question()
        summary_text = Helper().summarizer(context)
        print("I am under summary_text.")
        for wrp in wrap(summary_text, 150):
            print (wrp)
        # np = getnounphrases(summary_text,sentence_transformer_model,3)
        np =  Helper().get_keywords(context,summary_text)
        print("I am under np.")
        print ("\n\nNoun phrases",np)
        output=""
        for answer in np:
            ques1 = ques.get_question(summary_text, answer)
            if radiobutton=="Wordnet":
                distractors = get_distractors_wordnet(answer)
                # output= output + ques + "\n" + "Ans: "+answer.capitalize() + "\n\n"
                output = output + "<b style='color:blue;'>" + ques1 + "</b>"
                # output = output + "<br>"
                output = output + "<b style='color:green;'>" + "Ans: " +answer.capitalize()+  "</b>"
                if len(distractors)>0:
                    for distractor in distractors[:4]:
                        output = output + "<b style='color:brown;'>" + distractor+  "</b>"
                output = output + "<br>"
            else:
                distractors = get_distractors(answer.capitalize(), ques1, 40, 0.2)
                # output= output + ques + "\n" + "Ans: "+answer.capitalize() + "\n\n"
                output = output + "<b style='color:blue;'>" + ques1 + "</b>"
                # output = output + "<br>"
                output = output + "<b style='color:green;'>" + "Ans: " +answer.capitalize()+  "</b>"
                if len(distractors)>0:
                    for distractor in distractors[:4]:
                        output = output + "<b style='color:brown;'>" + distractor+  "</b>"
                output = output + "<br>"
        print("I am under")

        summary ="Summary: "+ summary_text
        for answer in np:
            summary = summary.replace(answer,"<b>"+answer+"</b>")
            summary = summary.replace(answer.capitalize(),"<b>"+answer.capitalize()+"</b>")
        output = output + "<p>"+summary+"</p>"
        return output
    except Exception as ex:
            raise ex
if __name__ == "__main__":
    iface = gr.Interface(fn=generate_question, inputs=[context, radiobutton], outputs=output).launch(debug=True, share=True)
    
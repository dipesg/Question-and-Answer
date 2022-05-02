from utils.helper import Helper
from updatedgetquestion import Question
from textwrap3 import wrap
import gradio as gr
context = gr.inputs.Textbox(lines=10, placeholder="Enter paragraph/content here...")
output = gr.outputs.HTML(  label="Question and Answers")

def generate_question(context):
    summary_text = Helper().summarizer(context)
    for wrp in wrap(summary_text, 150):
      print (wrp)
    np =  Helper().get_keywords(context,summary_text)
    print ("\n\nNoun phrases",np)
    output=""
    for answer in np:
      ques = Question().get_question(summary_text,answer)
      # output= output + ques + "\n" + "Ans: "+answer.capitalize() + "\n\n"
      output = output + "<b style='color:blue;'>" + ques + "</b>"
      # output = output + "<br>"
      output = output + "<b style='color:green;'>" + "Ans: " +answer.capitalize()+  "</b>"
      output = output + "<br>"

    summary ="Summary: "+ summary_text
    for answer in np:
      summary = summary.replace(answer,"<b>"+answer+"</b>")
      summary = summary.replace(answer.capitalize(),"<b>"+answer.capitalize()+"</b>")
    output = output + "<p>"+summary+"</p>"
    
    return output

if __name__ == "__main__":
    iface = gr.Interface(fn=generate_question, inputs=context, outputs=output).launch(debug=True, share=True)
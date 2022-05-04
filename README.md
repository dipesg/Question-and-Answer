# Question-Answering

This repository contains an implementation of the question-answering system. The main goal of the project is to learn working
with 🤗 transformers architecture **T5** with SQuAD 2.0 dataset.

The QA system is built using:
* HuggingFace's BERT transformer with custom head, fine-tuned on SQuAD v2.0, using only possible questions.

## Demo
![qna](https://user-images.githubusercontent.com/75604769/166700589-18e8d5ab-d70d-45a4-b50f-7796055da9cb.png)

## Installation and running

*  Clone the repository
```
git clone https://github.com/dipesg/Question-and-Answer.git
```

*  Create and activate conda environment:
```
conda create -n venv python = 3.9 -y
conda activate venv
```

* Run the app:
```
python app.py
```

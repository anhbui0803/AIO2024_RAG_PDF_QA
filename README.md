# AIO2024_RAG_PDF_QA

## Description
AIO2024 Module-1 project: Q&amp;A with RAG and Chainlit.

## How to use
Firstly, clone the repository to local:
```
$ git clone https://github.com/anhbui0803/AIO2024_RAG_PDF_QA.git
```
### Option 1: Using conda environment
1. Create new conda environment and install required dependencies:
```
$ conda create -n <env_name> -y python=3.11
$ conda activate <env_name>
$ pip3 install -r requirements.txt
```
2. Host chainlit app:
```
$ chainlit run app.py
```
### Option 2: Using pip virtualenv
1. Create new pip virtual environment and install required dependencies:

If you don't have virtualenv, please install via pip: 
```
$ python -m pip install --user virtualenv
```
```
$ virtualenv <env_name>
$ <env_name>\Scripts\activate
# python -m pip install -r requirements.txt
```
2. Host chainlit app: 
```
$ python -m chainlit run app.py
```

## Screenshot of the demo
![image](https://github.com/anhbui0803/AIO2024_RAG_PDF_QA/assets/94179304/1d65f230-4ae3-42cf-a7e6-4b9711632b83)

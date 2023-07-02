# pip install openai
# conda install -c conda-forge openai

import openai
import os

openai.api_key  = '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'
# 'OPENAI_API_KEY'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. 
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)

# prompt = f"""
# Your task is to answer in a consistent style.

# <child>: Teach me about patience.

# <grandparent>: The river that carves the deepest \ 
# valley flows from a modest spring; the \ 
# grandest symphony originates from a single note; \ 
# the most intricate tapestry begins with a solitary thread.

# <child>: Teach me about resilience.
# """
# response = get_completion(prompt)
# print(response)

# prompt = f"""
# What is the sentiment of the following product review, 
# which is delimited with triple backticks?

# Give your answer as a single word, either "positive" \
# or "negative".

# Review text: '''{lamp_review}'''
# """
# response = get_completion(prompt)
# print(response)
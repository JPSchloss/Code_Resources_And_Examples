# You may need to install the openai library and can do so with the code below.
# pip install openai
# conda install -c conda-forge openai

import openai

openai.api_key  = 'OPENAI_API_KEY_GOES_HERE'

# This fun
def helper_function(prompt, model="gpt-3.5-turbo"):
    # This assignes the role and the content to be passed to the OpenAI API. 
    messages = [{"role": "user", "content": prompt}]

    # This is the function to actually get a response from the API. 
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=100, # The max number of tokens to generate.
        echo=False, # Whether to return the prompt in addition to the generated completion
        temperature=0, # What sampling temperature to use, between 0 and 2, higher values will make the output more random. 
        #top_p=1.0, # An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability. 
        #frequency_penalty=0.0, # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text.
        #presence_penalty=0.0, # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        #stop=["//"] # Up to 4 sequences where the API will stop generating further tokens.
    )
    return response.choices[0].message["content"]

text = f"""
This is text that will be an example that the model \
can summarize. The model can do many things!
"""
# This is where you define the prompt to send to the model.
prompt = f"""
Summarize the text below into a single sentence.
```{text}```
"""
response = helper_function(prompt)
print(response)
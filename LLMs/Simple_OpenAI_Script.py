from openai import OpenAI

def summarize_text(text):
    client = OpenAI(
        api_key='YOUR API KEY GOES HERE', 
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Summarize the following text into a single sentence."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model="gpt-3.5-turbo",  
        )
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"An error occurred: {e}"


# Example usage
text = "Text to be summarized."
summary = summarize_text(text)
print("Summary:", summary)

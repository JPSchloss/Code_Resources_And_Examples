from bs4 import BeautifulSoup
import requests

def get_job_description_from_linkedin(public_url):
    # Pull the HTML from the URL location and parse it with Beautiful Soup.
    html = requests.get(public_url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = []
    
    # Find all of the description text in the Job Posting and loop through them to get the text. 
    for lines in soup.findAll('div', {'class': 'description__text'}):
        text.append(lines.get_text())

    # For each line in the text, strip the text, and join together using a newline. 
    lines = (line.strip() for line in text)
    text = '\n'.join(line for line in lines if line)
    
    return text

# Public LinkedIn Job Post URL to Scrape. 
public_url = 'https://www.linkedin.com/jobs/view/3545720466'

# Initiating Function
text = get_job_description_from_linkedin(public_url)
print(text)
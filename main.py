# LLMs
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Streamlit
import streamlit as st

# Scraping
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# YouTube
from langchain.document_loaders import YoutubeLoader
# !pip install youtube-transcript-api

# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()

# Get your API keys set
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet')

# Load up your LLM
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key, max_tokens=2000, model_name='gpt-4')
    return llm

# A function that will be called only if the environment's openai_api_key isn't set
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text


# Here we'll pull data from a website and return its text
def pull_from_website(url):
    st.write("Getting webpages...")
    # Doing a try in case it doesn't work
    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return
    
    # Put your response in a beautiful soup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get your text
    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)
     
    return text

# Pulling data from YouTube in text form
def get_video_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript

# Function to change our long text about a person into documents
def split_text(user_information):
    # First we make our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)

    # Then we split our user information into different documents
    docs = text_splitter.create_documents([user_information])

    return docs

# Prompts - We'll do a dynamic prompt based on the option the users selects
# We'll hold different instructions in this dictionary below
response_types = {
    'Interview Questions' : """
        Your goal is to generate interview questions that we can ask them
        Please respond with list of a few interview questions based on the topics above
    """,
    '1-Page Summary' : """
        Your goal is to generate a 1 page summary about them
        Please respond with a few short paragraphs that would prepare someone to talk to this person
    """
}

map_prompt = """You are a helpful AI bot that aids a user in research.
Below is information about a person named {persons_name}.
Information will include  interview transcripts and blog posts about {persons_name}
Use specifics from the research when possible

{response_type}

% START OF INFORMATION ABOUT {persons_name}:
{text}
% END OF INFORMATION ABOUT {persons_name}:

YOUR RESPONSE:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "persons_name", "response_type"])

combine_prompt = """
You are a helpful AI bot that aids a user in research.
You will be given information about {persons_name}.
Do not make anything up, only use information which is in the person's context

{response_type}

% PERSON CONTEXT
{text}

% YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "persons_name", "response_type"])

# Start Of Streamlit page
st.set_page_config(page_title="LLM Assisted Interview Prep", page_icon=":robot:")

# Start Top Information
st.header("LLM Assisted Interview Prep")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Have an interview coming up? Your subject is probably on YouTube or the web. This tool is meant to help you generate \
                interview questions based off of topics they've recently tweeted or talked about.\
                \n\nThis tool is powered by [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/#) [markdownify](https://pypi.org/project/markdownify/), [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
                [@GregKamradt](https://twitter.com/GregKamradt). \n\n View Source Code on [Github](https://github.com/gkamradt/globalize-text-streamlit/blob/main/main.py)")

with col2:
    st.image(image='Researcher.png', width=300, caption='Mid Journey: A researcher who is really good at their job and utilizes twitter to do research about the person they are interviewing. playful, pastels. --ar 4:7')
# End Top Information

st.markdown("## :older_man: Your LLM Researcher")

# Output type selection by the user
output_type = st.radio(
    "Output Type:",
    ('Interview Questions', '1-Page Summary'))

# Collect information about the person you want to research
person_name = st.text_input(label="Person's Name",  placeholder="Ex: Elad Gil", key="persons_name")
youtube_videos = st.text_input(label="YouTube URLs (Use , to seperate videos)",  placeholder="Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk, https://www.youtube.com/watch?v=c_hO_fjmMnk", key="youtube_user_input")
webpages = st.text_input(label="Web Page URLs (Use , to seperate urls. Must include https://)",  placeholder="https://eladgil.com/", key="webpage_user_input")


# Output
st.markdown(f"### {output_type}:")

# Get URLs from a string
def parse_urls(urls_string):
    """Split the string by comma and strip leading/trailing whitespaces from each URL."""
    return [url.strip() for url in urls_string.split(',')]

# Get information from those URLs
def get_content_from_urls(urls, content_extractor):
    """Get contents from multiple urls using the provided content extractor function."""
    return "\n".join(content_extractor(url) for url in urls)

button_ind = st.button("*Generate Output*", type='secondary', help="Click to generate output based on information")

# Checking to see if the button_ind is true. If so, this means the button was clicked and we should process the links
if button_ind:
    if not (youtube_videos or webpages):
        st.warning('Please provide links to parse', icon="⚠️")
        st.stop()

    if not OPENAI_API_KEY:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

    if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':
        # If the openai key isn't set in the env, put a text box out there
        OPENAI_API_KEY = get_openai_api_key()

    # Go get your data
    video_text = get_content_from_urls(parse_urls(youtube_videos), get_video_transcripts) if youtube_videos else ""
    website_data = get_content_from_urls(parse_urls(webpages), pull_from_website) if webpages else ""

    user_information = "\n".join([video_text, website_data])

    user_information_docs = split_text(user_information)

    # Calls the function above
    llm = load_LLM(openai_api_key=OPENAI_API_KEY)

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template,
                                 # verbose=True
                                 )
    
    st.write("Sending to LLM...")

    # Here we will pass our user information we gathered, the persons name and the response type from the radio button
    output = chain({"input_documents": user_information_docs, # The seven docs that were created before
                    "persons_name": person_name,
                    "response_type" : response_types[output_type]
                    })

    st.markdown(f"#### Output:")
    st.write(output['output_text'])

import streamlit as st
from dotenv import load_dotenv
from newsapi import NewsApiClient
from transformers import pipeline
import os

# Load environment variables
load_dotenv()

# Retrieve API keys
newsapi_key = os.getenv("NEWSAPI_KEY")

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=newsapi_key)

# Initialize Hugging Face's text generation pipeline
hf_generator = pipeline('text-generation', model='distilgpt2')

# Define the prompt template
template = """
You are an AI assistant helping an equity research analyst. Given the following query and the provided news article summaries, provide an overall summary.
Query: {query}
Summaries: {summaries}
"""

# Helper Functions
def get_news_articles(query):
    articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy")
    return articles["articles"]

def summarize_articles(articles):
    summaries = []
    for article in articles[:5]:  # Limit to top 5 articles
        summaries.append(article["description"] or "No description available")
    return " ".join(summaries)

def get_summary(query):
    articles = get_news_articles(query)
    article_summaries = summarize_articles(articles)
    prompt = template.format(query=query, summaries=article_summaries)
    response = hf_generator(prompt, max_length=300)
    return response[0]["generated_text"]

# Streamlit App
st.set_page_config(page_title="Equity Research News Tool", page_icon="📈", layout="wide")

# Add a banner image (img1.jpg) with reduced size
st.image('img1.jpeg', use_column_width=True, width=800)   

st.title("Equity Research News Tool")
st.write("Enter your query to get the latest news articles summarized")

# Query Input
query = st.text_input("Query", placeholder="e.g., Impact of inflation on stock markets")

if st.button("Get News"):
    if query:
        st.write("### Fetching Articles and Generating Summary...")
        try:
            summary = get_summary(query)
            st.write("### Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before clicking the button.")

# Footer with a bit of styling
st.write("---")
st.markdown("""
    <style>
        .footer {
            text-align: center;
            font-size: 12px;
            color: grey;
        }
    </style>
    <div class="footer">
        Powered by <a href="https://huggingface.co" target="_blank">Hugging Face</a>, 
        <a href="https://newsapi.org" target="_blank">NewsAPI</a>, and <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </div>
""", unsafe_allow_html=True)


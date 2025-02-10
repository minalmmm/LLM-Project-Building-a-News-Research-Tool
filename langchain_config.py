# Import Libraries

from dotenv import load_dotenv
import os
from newsapi import NewsApiClient
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys
newsapi_key = os.getenv("NEWSAPI_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=newsapi_key)

# Initialize Chat Groq LLM
groq_llm = ChatGroq(
    temperature=0,
    model = 'mixtral-8x7b-32768',
    groq_api_key = groq_api_key
    )

# Define the prompt template
template = """
You are an AI assistant helping an equity research analyst. Given 
the following query and the provided news article summaries, provide 
an overall summary.

Query: {query}
Summaries: {summaries}
"""
# Create the prompt
prompt = PromptTemplate.from_template(template)

# Combine prompt and LLM into a pipeline
pipeline = prompt | groq_llm

# Function to fetch news articles using NewsAPI
def get_news_articles(query):
    articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy")
    return articles["articles"]

# Function to summarize news articles
def summarize_articles(articles):
    summaries = []
    for article in articles[:5]:  # Limit to top 5 articles
        summaries.append(article["description"] or "No description available")
    return " ".join(summaries)

# Function to generate the final summary using Groq LLM
def get_summary(query):
    articles = get_news_articles(query)
    article_summaries = summarize_articles(articles)
    pipeline = prompt | groq_llm
    response = pipeline.invoke({"query": query, "summaries": article_summaries})
    return response

# Test the enhanced setup
if __name__ == "__main__":
    test_query = "Impact of inflation on stock markets"
    try:
        summary = get_summary(test_query)
        print("Generated Enhanced Summary:")
        print(summary)
    except Exception as e:
        print(f"Error: {e}")
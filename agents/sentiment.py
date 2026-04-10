from langchain.chat_models import ChatOpenAI
from config.env import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4", temperature=0.3)

def analyze_sentiment(text: str) -> str:
    prompt = f"""
Analyze the sentiment of the following text. Reply only with one word: 'positive', 'negative', or 'neutral'.

Text: \"\"\"{text}\"\"\"
"""
    response = llm.invoke(prompt)
    sentiment = response.content.strip().lower()
    if 'negative' in sentiment:
        return 'negative'
    elif 'neutral' in sentiment:
        return 'neutral'
    else:
        return 'positive'

from langchain_openai import ChatOpenAI
from config.env import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)

def reflect_answer(state):
    prompt = f"The initial answer is:\n\n{state.initial_response}\n\nEvaluate and improve this response if needed."
    improved = llm.invoke(prompt)
    return state.model_copy(update={"improved_response": improved.content})

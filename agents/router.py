from agents.sentiment import analyze_sentiment
from utils.vector_store import load_faiss_vectorstore
from pydantic import BaseModel

vectorstore = load_faiss_vectorstore()
departments = ['customer', 'hr', 'admin', 'billing']

class StateSchema(BaseModel):
    user_input: str
    dept_choice: str
    route: str = None
    sentiment: str = None
    message: str = None
    initial_response: str = None
    improved_response: str = None

def user_router(state: StateSchema) -> StateSchema:
    dept_choice = state.dept_choice.lower().strip()
    user_input = state.user_input

    if dept_choice == "other":
        return state.model_copy(update={'route': 'human_support', 'message': "Connecting to human support for other inquiries."})

    if dept_choice not in departments:
        # Invalid department, route to human
        return state.model_copy(update={'route': 'human_support', 'message': "Department not recognized, connecting to human support."})

    sentiment = analyze_sentiment(user_input)

    if sentiment == 'negative':
        return state.model_copy(update={'route': 'human_support', 'sentiment': sentiment, 'message': "Negative sentiment detected, connecting to human support."})

    # Positive or neutral sentiment, proceed with RAG
    return state.model_copy(update={'route': dept_choice, 'sentiment': sentiment})

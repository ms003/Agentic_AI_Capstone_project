from langgraph.graph import StateGraph, END
from agents.router import user_router
from agents.department_agent import rag_and_llm_subagent, reflect_on_answer
from agents.sentiment import analyze_sentiment
from agents.human_support import human_support_node
from utils.state import StateSchema
from utils.vector_store import load_faiss_vectorstore
from utils.markdown import log_to_markdown

import os

departments = ['customer', 'hr', 'admin', 'billing']

vectorstore = load_faiss_vectorstore()

workflow = StateGraph(state_schema=StateSchema)

workflow.add_node("user_router", user_router)

for dept in departments:
    workflow.add_node(dept, rag_and_llm_subagent)

workflow.add_node("human_support", human_support_node)

workflow.set_entry_point("user_router")

workflow.add_conditional_edges("user_router", lambda state: state.route, {
    "customer": "customer",
    "hr": "hr",
    "admin": "admin",
    "billing": "billing",
    "human_support": "human_support"
})

for dept in departments:
    workflow.add_edge(dept, END)
workflow.add_edge("human_support", END)

compiled_graph = workflow.compile()

def run_app():
    conversation_history = []

    while True:
        print("\nWelcome! Please select a department to connect with: customer, HR, admin, billing, or other")
        dept_choice = input("Department: ").strip().lower()

        if dept_choice == 'exit':
            print("Goodbye!")
            break

        while True:
            user_input = input("\nPlease enter your message (or type 'change' to select a different department, or 'exit' to quit):\n")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                log_to_markdown(conversation_history)  # Save on exit
                return
            if user_input.lower() == 'change':
                break

            state = StateSchema(dept_choice=dept_choice, user_input=user_input)
            result = compiled_graph.invoke(state)

            initial_response = result.get('initial_response', 'No response.')
            improved_response = result.get('improved_response', 'No improved response.')

            print("\nInitial response:")
            print(initial_response)

            print("\nImproved response:")
            print(improved_response)

            # Log the turn
            conversation_history.append({
                "department": dept_choice,
                "user": user_input,
                "initial": initial_response,
                "improved": improved_response
            })

            if result.get('route') == "human_support":
                break

if __name__ == "__main__":
    run_app()

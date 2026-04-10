from utils.state import StateSchema
from agents.department_agent import reflect_on_answer

def human_support_node(state: StateSchema) -> StateSchema:
    """
    Handles the human support node in the workflow.
    If a custom message is already set (e.g. by the router),
    it returns that message as both initial and improved response without reflection.
    Otherwise, it reflects on the default message before responding.
    """
    initial_msg = state.message or "Someone will be with you shortly."

    if state.message:
        # Custom message exists, do not reflect to avoid overwriting it
        improved_msg = initial_msg
    else:
        # No custom message, reflect on default message
        improved_msg = reflect_on_answer(state.user_input, "human_support", initial_msg)

    return state.model_copy(update={
        'initial_response': initial_msg,
        'improved_response': improved_msg,
    })

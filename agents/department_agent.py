from utils.state import StateSchema
from langchain.chat_models import ChatOpenAI
from utils.vector_store import load_faiss_vectorstore
from config.env import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from utils.vector_store import vectorstore
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.3)

def rag_and_llm_subagent(state):
    query = state.user_input
    dept = state.route

    # Retrieve docs with similarity scores
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    # Filter docs by department
    filtered_docs_with_scores = [
        (doc, score) for doc, score in retrieved_docs_with_scores
        if doc.metadata.get("type", "").lower() == dept.lower()
    ]

    # If no filtered docs, use top 3 overall
    if not filtered_docs_with_scores:
        filtered_docs_with_scores = retrieved_docs_with_scores[:3]
    else:
        filtered_docs_with_scores = filtered_docs_with_scores[:3]

    # Check similarity score threshold (assuming score is similarity here)
    max_score = max(score for _, score in filtered_docs_with_scores) if filtered_docs_with_scores else 0

    context = "\n".join(doc.page_content for doc, _ in filtered_docs_with_scores)

    if max_score >= 0.7:
        # Formulate response based on RAG only
        prompt = f"""
Using the following context information from the {dept} knowledge base, answer the user query specifically.

Context:
{context}

Question:
{query}
"""
        initial_answer = llm.invoke(prompt).content.strip()
        improved_answer = initial_answer
    else:
        # Use RAG + LLM with reflection
        prompt = f"""
Using the following context information from the {dept} knowledge base, answer the user query specifically.

Context:
{context}

Question:
{query}
"""
        initial_answer = llm.invoke(prompt).content.strip()

        # Reflect on the initial answer before finalizing
        improved_answer = reflect_on_answer(query, dept, initial_answer)

    return state.model_copy(update={
        'initial_response': initial_answer,
        'improved_response': improved_answer,
        'answer': improved_answer
    })

def reflect_on_answer(query, dept, answer):
    prompt = f"""
You are an assistant reflecting on a previously generated answer.

User query: "{query}"
Department: {dept}
Initial answer: "{answer}"

Please review the answer for clarity, relevance, and completeness. If the answer is good, respond with 'No changes needed.' Otherwise, provide an improved version of the answer.
"""
    reflection = llm.invoke(prompt).content.strip()
    if reflection.lower() == 'no changes needed.':
        return answer
    else:
        return reflection

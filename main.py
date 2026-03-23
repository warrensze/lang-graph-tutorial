from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model(
    "llama3.1:8b",
    model_provider="ollama",
    temperature=0,  # controls randomness
    base_url="http://localhost:11434",  # default local Ollama URL
)


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response.",
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message = state["messages"][-1]
    classifer_llm = llm.with_structured_output(MessageClassifier)

    result = classifer_llm.invoke(
        [
            {
                "role": "system",
                "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """,
            },
            {"role": "user", "content": last_message.content},
        ]
    )
    return {"message_type": result.message_type}


def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked.""",
        },
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.""",
        },
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)
graph_builder.add_edge(START, end_key="classifier")
graph_builder.add_edge(start_key="classifier", end_key="router")
graph_builder.add_conditional_edges(
    source="router",
    path=lambda state: state.get("next"),
    path_map={"therapist": "therapist", "logical": "logical"},
)

graph_builder.add_edge(start_key="therapist", end_key=END)
graph_builder.add_edge(start_key="logical", end_key=END)

graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}
    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()

"""
user_input = input("Enter a message: ")
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print(state["messages"])
print(state["messages"][-1].content)
"""

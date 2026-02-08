import os
from dotenv import load_dotenv

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from graph.graph import app

load_dotenv()


def run_cli(app):
    """
    Simple CLI harness to test LangGraph interrupt/resume logic.
    """
    checkpointer = MemorySaver()  # needed for resume to work
    # If you compile outside, ensure you used this checkpointer there.
    # app = graph.compile(checkpointer=checkpointer)

    thread_id = "local-test-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("RAG QA CLI (type 'exit' to quit)\n")

    while True:
        user_q = input("You: ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            break

        # First pass
        state = {"question": user_q, "game_type": None}
        result = app.invoke(state, config=config)

        # Handle 0+ interrupts (your graph may interrupt more than once)
        while "__interrupt__" in result:
            intr = result["__interrupt__"][0]
            # intr.value is usually the message you passed to interrupt(...)
            print(f"\nAssistant (clarify): {intr.value}")
            user_reply = input("You (clarification): ").strip()

            # Resume the graph with user's clarification
            result = app.invoke(Command(resume=user_reply), config=config)

        # At this point, graph finished (or routed to END)
        generation = result.get("generation")
        if generation:
            print(f"\nAssistant: {generation}\n")
        else:
            # Helpful debug output if you didn't store answer under "answer"
            print("\nAssistant: (No 'answer' field found in state.)")
            print("Final state keys:", list(result.keys()), "\n")


if __name__ == "__main__":
    run_cli(app)

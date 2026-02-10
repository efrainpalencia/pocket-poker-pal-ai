from langgraph.types import Command
from graph.graph import graph


def cli_run():
    thread_id = "local-test-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("QA Test (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        result = graph.invoke(
            {"question": q, "game_type": None}, config=config)

        # Handle 0+ interrupts (game_type clarify OR later retry_or_clarify)
        while "__interrupt__" in result:
            intr = result["__interrupt__"][0]
            print(f"\nAssistant (clarify): {intr.value}")
            reply = input("You (reply): ").strip()
            result = graph.invoke(Command(resume=reply), config=config)

        # Print final answer + useful debug fields
        print("\nAssistant:", result.get("generation"))
        print("Debug:", {
            "game_type": result.get("game_type"),
            "grounded": result.get("grounded"),
            "attempts": result.get("attempts", 0),
            "docs": len(result.get("documents", [])),
        })
        print()

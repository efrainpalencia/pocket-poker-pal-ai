from __future__ import annotations

from uuid import uuid4

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

# plus close_checkpointer if you added it
from graph.checkpointer import build_checkpointer
from graph.graph import build_graph


def _ask_cli_game_type() -> str:
    while True:
        gt = input("CLI ruleset [tournament/cash-game]: ").strip().lower()
        if gt in {"tournament", "cash-game"}:
            return gt
        print("Please enter 'tournament' or 'cash-game'.")


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt(out: dict) -> dict | None:
    """Match your backend interrupt style: {'__interrupt__': [Interrupt(value=...)]}"""
    if not (isinstance(out, dict) and "__interrupt__" in out and out["__interrupt__"]):
        return None

    intr = out["__interrupt__"][0]
    value = getattr(intr, "value", None)

    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        return {"type": "free_text", "message": value.strip()}

    return {"type": "free_text", "message": "Can you clarify?"}


def cli_run(thread_id: str | None = None) -> None:
    """
    Local CLI runner for graph QA.

    - Builds the graph using the same production compilation path.
    - Uses the graph's interrupt protocol (no forced state mutations).
    - CLI prompts for ruleset each question, but only submits it when the graph asks.
    """
    # Build checkpointer + graph (same architecture as FastAPI lifespan)
    checkpointer = build_checkpointer() or MemorySaver()
    graph = build_graph(checkpointer)

    thread_id = thread_id or f"cli-{uuid4()}"
    config = _cfg(thread_id)

    print(f"QA Test — thread_id={thread_id} (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # Your preference: ask ruleset up front (but don't inject into question)
        cli_game_type = _ask_cli_game_type()

        # Run the graph
        out = graph.invoke({"question": q}, config=config)

        # Handle 0+ interrupts
        max_loops = 8
        loops = 0

        while True:
            prompt = _extract_interrupt(out)
            if not prompt:
                break  # no interrupt

            loops += 1
            if loops > max_loops:
                print(
                    "\nAssistant: Too many clarification loops; stopping (safety cap).")
                break

            # Show prompt message nicely
            msg = prompt.get("message") if isinstance(prompt, dict) else None
            if msg:
                print(f"\nAssistant (clarify): {msg}")
            else:
                print(f"\nAssistant (clarify): {prompt}")

            # If the prompt is choose_ruleset, let Enter accept CLI selection
            reply: str
            if isinstance(prompt, dict) and prompt.get("type") == "choose_ruleset":
                reply = input(
                    f"You (reply) [Enter = {cli_game_type}]: ").strip().lower()
                if not reply:
                    reply = cli_game_type
            else:
                reply = input("You (reply): ").strip()

            out = graph.invoke(Command(resume=reply), config=config)

        # Final output
        gen = out.get("generation") if isinstance(out, dict) else None
        if gen:
            print("\nAssistant:", gen)
        else:
            print("\nAssistant: (no generation returned)")

        # Debug summary (optional)
        if isinstance(out, dict):
            print(
                "Debug:",
                {
                    "game_type": out.get("game_type"),
                    "namespace": out.get("namespace"),
                    "grounded": out.get("grounded"),
                    "confidence": out.get("confidence"),
                    "retry_count": out.get("retry_count", 0),
                    "docs": len(out.get("documents", []) or []),
                    "force_end": out.get("force_end"),
                },
            )
        print()

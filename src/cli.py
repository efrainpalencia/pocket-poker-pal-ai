from langgraph.types import Command
from graph.graph import graph


def _ask_cli_game_type() -> str:
    while True:
        gt = input("CLI ruleset [tournament/cash-game]: ").strip().lower()
        if gt in {"tournament", "cash-game"}:
            return gt
        print("Please enter 'tournament' or 'cash-game'.")


def cli_run():
    thread_id = "local-test-2"
    config = {"configurable": {"thread_id": thread_id}}

    print("QA Test (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # 1) CLI asks every time (your requirement)
        cli_game_type = _ask_cli_game_type()

        # 2) Force the graph to ALSO ask every time (redundancy)
        #    We do this by explicitly setting needs_clarification=True so after_route -> retry_or_clarify.
        #    Also clear sticky flags so old state can’t short-circuit.
        result = graph.invoke(
            {
                "question": q,

                # ✅ wipe persisted routing so route_or_clarify cannot short-circuit
                "game_type": None,
                "namespace": None,
                "meta_filter": {},

                # ✅ force graph to ask ruleset every turn (your redundancy)
                "needs_clarification": True,
                "missing_info": ["Is this about tournament rules or cash-game rules?"],

                # ✅ clear sticky control flags
                "force_end": False,
                "retry_count": 0,

                # optional: clear turn artifacts
                "documents": [],
                "generation": "",
                "generation_structured": {},
                "context_used": "",
                "confidence": 0.0,
                "grounded": False,
            },
            config=config,
        )

        # Handle 0+ interrupts
        max_interrupts = 8
        interrupts_seen = 0

        while "__interrupt__" in result:
            interrupts_seen += 1
            if interrupts_seen > max_interrupts:
                print(
                    "\nAssistant: Stopping due to too many clarification loops (debug safety cap).")
                break

            intr = result["__interrupt__"][0]
            payload = intr.value
            print(f"\nAssistant (clarify): {payload}")

            # If this is the ruleset prompt, allow Enter to accept the CLI selection
            if isinstance(payload, dict) and payload.get("type") == "choose_ruleset":
                reply = input(
                    f"You (reply) [Enter = {cli_game_type}]: ").strip().lower()
                if not reply:
                    reply = cli_game_type
            else:
                reply = input("You (reply): ").strip()

            result = graph.invoke(Command(resume=reply), config=config)

        print("\nAssistant:", result.get("generation"))
        print(
            "Debug:",
            {
                "game_type": result.get("game_type"),
                "namespace": result.get("namespace"),
                "grounded": result.get("grounded"),
                "confidence": result.get("confidence"),
                "retry_count": result.get("retry_count", 0),
                "docs": len(result.get("documents", []) or []),
                "force_end": result.get("force_end"),
            },
        )
        print()

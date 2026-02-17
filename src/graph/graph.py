from graph.workflow import build_workflow


def build_graph(checkpointer):
    workflow = build_workflow()
    return workflow.compile(checkpointer=checkpointer)

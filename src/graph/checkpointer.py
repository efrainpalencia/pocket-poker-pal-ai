import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


def build_checkpointer():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        # Local dev fallback
        return MemorySaver()

    from langgraph.checkpoint.postgres import PostgresSaver

    saver = PostgresSaver.from_conn_string(database_url)
    saver.setup()  # creates tables if not exist
    return saver

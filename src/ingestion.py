import os
import re
import hashlib
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from graph.consts import SEMINOLE_NAMESPACE, TDA_NAMESPACE

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")
TOURNAMENT_FILE_PATH = os.getenv("TOURNAMENT_FILE_PATH")
CASH_GAME_FILE_PATH = os.getenv("CASH_GAME_FILE_PATH")

if not INDEX_NAME:
    raise RuntimeError("Missing INDEX_NAME env var")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# Fallback splitter for large blocks
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ---------------------------
# Helpers
# ---------------------------


def stable_id(rulebook: str, namespace: str, page: int, block_id: str, chunk_index: int, content: str) -> str:
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    safe_block = re.sub(r"[^a-zA-Z0-9_-]+", "_", block_id)[:60] or "block"
    return f"{rulebook}__{namespace}__p{page}__{safe_block}__c{chunk_index}__{h}"


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_tda_blocks(page_text: str) -> List[Tuple[str, str]]:
    """
    Split TDA content into blocks by:
      - Rule <number>:
      - RP-<number>:
    Returns list of (block_id, block_text).
    """
    text = page_text
    pattern = re.compile(r"(?m)^(Rule\s+\d+\:|RP-\d+\:)", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if not matches:
        cleaned = normalize_ws(text)
        return [("page", cleaned)] if cleaned else []

    blocks: List[Tuple[str, str]] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        first_line = block.splitlines()[0].strip()
        block_id = first_line.replace(" ", "_").replace(":", "")
        block_text = normalize_ws(block)

        if block_text:
            blocks.append((block_id, block_text))

    return blocks


def split_seminole_sections(page_text: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Split Seminole content into blocks by Section headers:
      - Section A
      - Section B
      ...
      - Section K
    Returns list of (section_id, section_text, section_title_line).
    """
    text = page_text
    pattern = re.compile(r"(?m)^(Section\s+[A-K]\b.*)$", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if not matches:
        cleaned = normalize_ws(text)
        return [("page", cleaned, None)] if cleaned else []

    blocks: List[Tuple[str, str, Optional[str]]] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        header_line = m.group(1).strip()
        sec_match = re.match(r"(?i)Section\s+([A-K])\b", header_line)
        sec_letter = sec_match.group(1).upper() if sec_match else "X"
        section_id = f"Section_{sec_letter}"
        section_title = header_line

        block_text = normalize_ws(block)
        if block_text:
            blocks.append((section_id, block_text, section_title))

    return blocks


def docs_from_blocks(
    blocks: List[Tuple],
    base_meta: dict,
    rulebook_name: str,
    namespace: str,
) -> Tuple[List[Document], List[str]]:
    """
    Convert blocks into Documents. If a block is too large, fallback-split it.
    Returns (documents, ids).
    """
    out_docs: List[Document] = []
    out_ids: List[str] = []

    for block in blocks:
        if len(block) == 2:
            block_id, block_text = block
            section_title = None
        else:
            block_id, block_text, section_title = block

        if not block_text:
            continue

        block_doc = Document(
            page_content=block_text,
            metadata=pinecone_safe_meta({
                **base_meta,
                "block_id": block_id,
                "section_title": section_title,  # will be removed if None
            }),

        )

        # If it's still too big, split further
        if len(block_text) > 1600:
            sub_docs = fallback_splitter.split_documents([block_doc])
        else:
            sub_docs = [block_doc]

        for j, sd in enumerate(sub_docs):
            page = sd.metadata.get("page", -1)
            sid = stable_id(
                rulebook=rulebook_name,
                namespace=namespace,
                page=page,
                block_id=block_id,
                chunk_index=j,
                content=sd.page_content,
            )
            sd.metadata["chunk_index"] = j
            out_docs.append(sd)
            out_ids.append(sid)

    return out_docs, out_ids


def add_in_batches(vs: PineconeVectorStore, docs: List[Document], ids: List[str], namespace: str, batch_size: int = 100):
    for start in range(0, len(docs), batch_size):
        end = start + batch_size
        vs.add_documents(
            documents=docs[start:end],
            ids=ids[start:end],
            namespace=namespace,
        )


def pinecone_safe_meta(meta: dict) -> dict:
    """
    Pinecone does not allow null metadata values.
    Remove keys with None and normalize list values to list[str] if present.
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        clean[k] = v
    return clean


# ---------------------------
# Ingestion functions
# ---------------------------


def ingest_tda_pdf(file_path: str):
    rulebook_name = "tda_2024"
    namespace = TDA_NAMESPACE

    if not file_path or not os.path.exists(file_path):
        raise RuntimeError(f"Invalid TOURNAMENT_FILE_PATH: {file_path}")

    print(f"📥 Ingesting {rulebook_name} into namespace='{namespace}'...")

    loader = PyPDFLoader(file_path)
    page_docs = loader.load()

    all_docs: List[Document] = []
    all_ids: List[str] = []

    for pdoc in page_docs:
        page = pdoc.metadata.get("page", -1)
        text = (pdoc.page_content or "").strip()
        if not text:
            continue

        blocks = split_tda_blocks(text)

        base_meta = {
            "rulebook": rulebook_name,
            "source_pdf": "TDA_2024",
            "namespace": namespace,
            "game_type": "tournament",
            "page": page,
            "source_file": os.path.basename(file_path),
        }

        docs, ids = docs_from_blocks(
            blocks, base_meta, rulebook_name, namespace)
        all_docs.extend(docs)
        all_ids.extend(ids)

    add_in_batches(vectorstore, all_docs, all_ids, namespace=namespace)
    print(f"✅ Ingested {len(all_docs)} chunks for {rulebook_name}")


def ingest_seminole_pdf(file_path: str):
    rulebook_name = "seminole_2025"
    namespace = SEMINOLE_NAMESPACE

    if not file_path or not os.path.exists(file_path):
        raise RuntimeError(f"Invalid CASH_GAME_FILE_PATH: {file_path}")

    print(f"📥 Ingesting {rulebook_name} into namespace='{namespace}'...")

    loader = PyPDFLoader(file_path)
    page_docs = loader.load()

    all_docs: List[Document] = []
    all_ids: List[str] = []

    for pdoc in page_docs:
        page = pdoc.metadata.get("page", -1)
        text = (pdoc.page_content or "").strip()
        if not text:
            continue

        blocks = split_seminole_sections(text)

        for block in blocks:
            section_id, section_text, section_title = block

            # 🔥 Refined game_type classification
            if section_id.lower() == "section_j":
                game_type = "tournament"
            else:
                game_type = "cash-game"

            base_meta = pinecone_safe_meta({
                "rulebook": rulebook_name,
                "source_pdf": "SEMINOLE_2025",
                "namespace": namespace,
                "game_type": game_type,
                "section": section_id,
                "section_title": section_title,
                "page": page,
                "source_file": os.path.basename(file_path),
            })

            docs, ids = docs_from_blocks(
                blocks=[block],
                base_meta=base_meta,
                rulebook_name=rulebook_name,
                namespace=namespace,
            )

            all_docs.extend(docs)
            all_ids.extend(ids)

    add_in_batches(vectorstore, all_docs, all_ids, namespace=namespace)
    print(f"✅ Ingested {len(all_docs)} chunks for {rulebook_name}")


# ---------------------------
# Run ingestion
# ---------------------------
ingest_tda_pdf(TOURNAMENT_FILE_PATH)
ingest_seminole_pdf(CASH_GAME_FILE_PATH)

print("🎯 Ingestion complete.")

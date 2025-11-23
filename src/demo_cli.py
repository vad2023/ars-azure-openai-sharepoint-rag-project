import logging

from .config import Config
from .sharepoint_client import SharePointClient
from .rag_pipeline import AzureOpenAIRag, simple_text_from_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = Config
    cfg.validate()

    sp = SharePointClient(cfg)
    site = sp.get_site()
    drive = sp.get_drive_for_library(site["id"])
    files = sp.list_files(drive["id"])

    docs = []
    for f in files:
        filename = f.get("name")
        logger.info("Downloading: %s", filename)
        data = sp.download_file_content(drive["id"], f["id"])
        text = simple_text_from_bytes(filename, data)
        docs.append((filename, text))

    rag = AzureOpenAIRag(cfg)
    rag.build_index_for_docs(docs)

    print("Index built. Ask questions about your SharePoint content (type 'exit' to quit).")
    while True:
        q = input("\n>>> Question: ")
        if not q or q.strip().lower() in {"exit", "quit"}:
            break
        answer = rag.answer_question(q)
        print("\n[Answer]\n")
        print(answer)


if __name__ == "__main__":
    main()

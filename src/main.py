import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from uvicorn.config import LOGGING_CONFIG

from llm import LLMRouter
from pdf_utils import unzip_pdf, get_pdf_page_count
from zotero_utils import (
    get_todo_items,
    get_items_needing_tags,
    get_items_without_tags,
    update_item_tags,
    write_note,
    get_pdf_children,
    ZOTERO_TAGS,
)

# Setup
app = FastAPI()
logger = logging.getLogger("Athena")


def setup_logger() -> None:
    """Setup application logger."""
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("application.log"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def summarize_and_tag_single_doc(item: dict, *, llm: LLMRouter, add_summary: bool) -> None:
    """Process a single document for summarization and/or tagging."""
    key = item["data"]["key"]
    logger.info(f"Handling item {key}")
    
    title = item["data"].get("title")
    if not title:
        logger.warning(f"Skipping item {key} because it has no title")
        update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
        return

    # Get PDF attachment
    pdf_items = get_pdf_children(key)
    if not pdf_items:
        logger.error(f"No PDF attachment found for item {key}, skipping.")
        update_item_tags(key, tags_to_add=[ZOTERO_TAGS["DENY"]])
        return

    # Extract PDF from zip
    project_root = Path(__file__).parent.parent
    pdf_path = unzip_pdf(project_root / "zotero" / f"{pdf_items[0]['key']}.zip")
    if not pdf_path:
        logger.error(f"Could not find a PDF for item {key} in the path, skipping.")
        update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
        return

    # Check PDF page count
    page_count = get_pdf_page_count(pdf_path)
    if not 5 <= page_count <= 100:
        logger.error(f"PDF length ({page_count} pages) is out of bounds, skipping.")
        update_item_tags(
            key,
            tags_to_add=[ZOTERO_TAGS["DENY"]],
            tags_to_remove=[ZOTERO_TAGS["TODO"]],
        )
        return

    # Process with LLM
    if add_summary:
        summary, tags = llm.summary_and_tags(pdf_path)
        if not summary:
            logger.error(f"Could not summarize item {key}, skipping.")
            update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
            return
        
        write_note(key, f"Summary\n\n{summary}")
        
        # Update tags
        tags_to_add = [ZOTERO_TAGS["SUMMARIZED"]]
        if tags:
            tags_to_add.extend(tags)
        
        update_item_tags(
            key,
            tags_to_add=tags_to_add,
            tags_to_remove=[ZOTERO_TAGS["TODO"]],
        )
    else:
        # Tags only
        tags = llm.tags_only(pdf_path)
        tags_to_add = [ZOTERO_TAGS["SUMMARIZED"]]
        if tags:
            tags_to_add.extend(tags)
        
        update_item_tags(
            key,
            tags_to_add=tags_to_add,
            tags_to_remove=[ZOTERO_TAGS["TODO"]],
        )


def summarize_and_tag_all_docs() -> None:
    """Summarize and tag all documents marked as TODO."""
    items = get_todo_items(limit=50)
    logger.info(f"Found {len(items)} items to summarize")
    
    llm = LLMRouter(os.getenv("SUMMARY_MODEL", "gemini-2.5-pro-exp-03-25"))
    
    for item in items:
        summarize_and_tag_single_doc(item, llm=llm, add_summary=True)


def add_missing_tags() -> None:
    """Add tags to items that need them."""
    items = get_items_needing_tags(limit=50)
    logger.info(f"Found {len(items)} items to tag")
    
    llm = LLMRouter(os.getenv("TAG_MODEL", "gemini-2.0-flash"))
    
    for item in items:
        summarize_and_tag_single_doc(item, llm=llm, add_summary=False)


def add_initial_tags() -> None:
    """Add initial TODO tags to items without any Athena tags."""
    items = get_items_without_tags(limit=50)
    for item in items:
        update_item_tags(item["data"]["key"], tags_to_add=[ZOTERO_TAGS["TODO"]])


# FastAPI endpoints
@app.get("/add_initial_tags/")
def fastapi_add_initial_tags():
    """Add initial TODO tags to untagged items."""
    try:
        add_initial_tags()
        return {"status": "Tags added successfully!"}
    except Exception as e:
        logger.error(f"Error adding initial tags: {e}")
        return {"status": "Error", "message": str(e)}


@app.get("/add_missing_tags/")
def fastapi_add_missing_tags(background_tasks: BackgroundTasks):
    """Add missing tags to items that need them."""
    background_tasks.add_task(add_missing_tags)
    return {"status": "Tags added successfully!"}


@app.get("/summarize/")
def summarize(background_tasks: BackgroundTasks):
    """Summarize and tag all TODO items."""
    background_tasks.add_task(summarize_and_tag_all_docs)
    return {"status": "Summary started"}


if __name__ == "__main__":
    setup_logger()
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = (
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    LOGGING_CONFIG["handlers"]["default"]["stream"] = sys.stdout
    LOGGING_CONFIG["loggers"]["uvicorn"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    LOGGING_CONFIG["loggers"]["uvicorn.error"] = {"level": "INFO"}
    LOGGING_CONFIG["loggers"]["uvicorn.access"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    uvicorn.run(app, host="0.0.0.0", port=5000, log_config=LOGGING_CONFIG)

import configparser
import logging
import re
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import uvicorn
from anthropic import Anthropic
from fastapi import BackgroundTasks, FastAPI
from pypdf import PdfReader
from pyzotero import zotero
from uvicorn.config import LOGGING_CONFIG

config = configparser.ConfigParser()
config.read("config.ini")

MODEL_NAME = config.get("general", "model_name")
ZOTERO_API_KEY = config.get("zotero", "api_key")
ZOTERO_USER_ID = config.getint("zotero", "user_id")
ZOTERO_TAGS = {
    "TODO": config.get("zotero", "todo_tag_name"),
    "SUMMARIZED": config.get("zotero", "summarized_tag_name"),
    "DENY": config.get("zotero", "deny_tag_name"),
    "ERROR": config.get("zotero", "error_tag_name")
}
FILE_BASE_PATH = config.get("zotero", "file_path")
CLAUDE_API_KEY = config.get("claude", "api_key")

app = FastAPI()
zot = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
logger = logging.getLogger("Clautero")


def setup_logger():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler("application.log")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def unzip_pdf(zip_file_name: str) -> Optional[str]:
    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
        pdf_files = [name for name in zip_ref.namelist() if name.endswith(".pdf")]
        if not pdf_files:
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp.write(zip_ref.read(pdf_files[0]))
            return temp.name


def extract_summary_and_tags(text: str) -> tuple[Optional[str], List[str]]:
    summary_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    tags_match = re.search(r'<tags>(.*?)</tags>', text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else None
    tags = []
    if tags_match:
        tag_text = tags_match.group(1).strip()
        raw_tags = re.split(r'[,\n]+', tag_text)
        for tag in raw_tags:
            cleaned_tag = re.sub(r'^[\s•-]+', '', tag.strip())
            if cleaned_tag:  # Only add non-empty tags
                tags.append(cleaned_tag)
    return summary, tags


def write_note(parent_id: str, note_text: str) -> None:
    template = zot.item_template("note")
    template["tags"] = [{"tag": MODEL_NAME}, {"tag": ZOTERO_TAGS["SUMMARIZED"]}]
    template["note"] = note_text.replace("\n", "<br>")
    zot.create_items([template], parent_id)


def update_item_tags(item_id: str, tags_to_add: List[str] = None, tags_to_remove: List[str] = None) -> None:
    item = zot.item(item_id)
    tags = item["data"]["tags"]
    tags = [tag for tag in tags if tag.get("tag") not in (tags_to_remove or [])]
    tags.extend({"tag": tag} for tag in (tags_to_add or []))
    item["data"]["tags"] = tags
    zot.update_item(item)


def extract_text_from_pdf(pdf_data: bytes) -> Optional[str]:
    try:
        return "".join(page.extract_text() for page in PdfReader(BytesIO(pdf_data)).pages)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None


def get_summary_and_tags(pdf_path: str) -> tuple[Optional[str], List[str]]:
    client = Anthropic(api_key=CLAUDE_API_KEY)
    try:
        with open("prompt.txt", "r") as f, open(pdf_path, "rb") as pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file.read())
            message = client.messages.create(
                system=f.read(),
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": pdf_text
                            }
                        ]
                    },
                    {
                        "role": "assistant",  # Pre-fill responses to circumvent refusals
                        "content": [
                            {
                                "type": "text",
                                "text": "<summary>"
                            }
                        ]
                    }

                ],
                model=MODEL_NAME,
            )
        # Add pre-filled part back in for valid tag
        return extract_summary_and_tags("<summary>\n" + message.content[0].text)
    except Exception as e:
        logger.error(f"Error summarizing and tagging {pdf_path}: {e}")
        return None, []


def summarize_all_docs():
    items = zot.top(tag=[ZOTERO_TAGS["TODO"], f"-{ZOTERO_TAGS["ERROR"]}", f"-{ZOTERO_TAGS["DENY"]}"], limit=50)
    logger.info(f"Found {len(items)} items to summarize")

    for item in items:
        key = item["data"]["key"]
        if not item["data"].get("title"):
            logger.warning(f"Skipping item {key} because it has no title")
            update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
            continue

        pdf_items = [child for child in zot.children(key)
                     if child.get("data", {}).get("contentType") == "application/pdf"]
        if not pdf_items:
            logger.error(f"No PDF attachment found for item {key}, skipping.")
            update_item_tags(key, tags_to_add=[ZOTERO_TAGS["DENY"]])
            continue

        pdf_path = unzip_pdf(str(Path(FILE_BASE_PATH) / f"{pdf_items[0]['key']}.zip"))
        if not pdf_path:
            logger.error(f"Could not find a PDF for item {key} in the path, skipping.")
            update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
            continue

        pdf_reader = PdfReader(pdf_path)
        if not 5 <= len(pdf_reader.pages) <= 100:
            logger.error("PDF length is out of bounds, skipping.")
            update_item_tags(key, tags_to_add=[ZOTERO_TAGS["DENY"]], tags_to_remove=[ZOTERO_TAGS["TODO"]])
            continue

        summary, tags = get_summary_and_tags(pdf_path)
        if not summary:
            logger.error(f"Could not summarize item {key}, skipping.")
            update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
            continue

        write_note(key, f"Summary\n\n{summary}")
        update_item_tags(key, tags_to_add=[ZOTERO_TAGS["SUMMARIZED"]] + tags, tags_to_remove=[ZOTERO_TAGS["TODO"]])


def add_missing_tags():
    items = zot.top(tag=[f"-{tag}" for tag in ZOTERO_TAGS.values()], limit=50)
    for item in items:
        update_item_tags(item["data"]["key"], tags_to_add=[ZOTERO_TAGS["TODO"]])


@app.get("/add_missing_tags/")
def fastapi_add_missing_tags():
    try:
        add_missing_tags()
        return {"status": "Tags added successfully!"}
    except Exception as e:
        return {"status": "Error", "message": str(e)}


@app.get("/summarize/")
def summarize(background_tasks: BackgroundTasks):
    background_tasks.add_task(summarize_all_docs)
    return {"status": "started summary"}


if __name__ == "__main__":
    setup_logger()
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    LOGGING_CONFIG["handlers"]["default"]["stream"] = sys.stdout
    LOGGING_CONFIG["loggers"]["uvicorn"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    LOGGING_CONFIG["loggers"]["uvicorn.error"] = {
        "level": "INFO",
    }
    LOGGING_CONFIG["loggers"]["uvicorn.access"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    uvicorn.run(app, host="0.0.0.0", port=5000, log_config=LOGGING_CONFIG)

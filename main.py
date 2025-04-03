import logging
import os
import random
import re
import sys
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

from google import genai
from google.genai import types, errors
import uvicorn
from anthropic import Anthropic
from fastapi import BackgroundTasks, FastAPI
from pypdf import PdfReader
from pyzotero import zotero
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from uvicorn.config import LOGGING_CONFIG

# Load configuration from environment variables
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
if not ZOTERO_API_KEY:
    raise ValueError("ZOTERO_API_KEY environment variable is required.")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
if not ZOTERO_USER_ID:
    raise ValueError("ZOTERO_USER_ID environment variable is required.")
ZOTERO_USER_ID = int(ZOTERO_USER_ID)
ZOTERO_TAGS = {
    "TODO": os.getenv("ZOTERO_TODO_TAG_NAME", "TODO"),
    "SUMMARIZED": os.getenv("ZOTERO_SUMMARIZED_TAG_NAME", "SUMMARIZED"),
    "DENY": os.getenv("ZOTERO_DENY_TAG_NAME", "DENY"),
    "ERROR": os.getenv("ZOTERO_ERROR_TAG_NAME", "ERROR"),
}

# Initialize application and services
app = FastAPI()
zot = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
logger = logging.getLogger("Athena")


def setup_logger() -> None:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("application.log"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def unzip_pdf(zip_file_path: Path) -> Path | None:
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        pdf_files = [name for name in zip_ref.namelist() if name.endswith(".pdf")]
        if not pdf_files:
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(zip_ref.read(pdf_files[0]))
            return Path(temp_file.name)


def extract_summary(text: str) -> str | None:
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_tags(text: str) -> List[str]:
    match = re.search(r"<tags>(.*?)</tags>", text, re.DOTALL)
    if not match:
        return []
    tag_text = match.group(1).strip()
    raw_tags = re.split(r"[,\n]+", tag_text)
    return [re.sub(r"^[\s•-]+", "", tag.strip()) for tag in raw_tags if tag.strip()]


def write_note(parent_id: str, note_text: str) -> None:
    template = zot.item_template("note")
    summary_model = os.getenv("SUMMARY_MODEL", "claude-3-5-sonnet-20240620")
    template["tags"] = [{"tag": summary_model}, {"tag": ZOTERO_TAGS["SUMMARIZED"]}]
    template["note"] = note_text.replace("\n", "<br>")
    zot.create_items([template], parent_id)


def update_item_tags(
    item_id: str,
    tags_to_add: List[str] | None = None,
    tags_to_remove: List[str] | None = None,
) -> None:
    item = zot.item(item_id)
    current_tags = {tag["tag"] for tag in item["data"]["tags"]}
    if tags_to_remove:
        current_tags.difference_update(tags_to_remove)
    if tags_to_add:
        current_tags.update(tags_to_add)
    item["data"]["tags"] = [{"tag": tag} for tag in current_tags]
    zot.update_item(item)


def extract_text_from_pdf(pdf_data: bytes) -> str | None:
    try:
        pdf_reader = PdfReader(BytesIO(pdf_data))
        text_pages = [page.extract_text() or "" for page in pdf_reader.pages]
        text = "".join(text_pages).strip()
        return text if text else None
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None


def run_claude_prompt(
    pdf_path: Path,
    system_prompt: str,
    prompt: str,
    model: str,
    prefilling: str,
) -> str | None:
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        raise ValueError("CLAUDE_API_KEY environment variable is required.")
    client = Anthropic(api_key=claude_api_key)
    with pdf_path.open("rb") as pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file.read())
    if not pdf_text:
        logger.error("No text extracted from PDF.")
        return None
    user_prompt = f"{prompt}\n\n<paper>\n{pdf_text}\n</paper>"
    try:
        message = client.messages.create(
            system=system_prompt,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": prefilling}],
                },
            ],
            model=model,
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Error running Anthropic prompt: {e}")
        return None


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def generate_content_with_retry(
    client: genai.Client,
    model_name: str,
    system_prompt: str,
    uploaded_file: types.File,
    prompt: str,
):
    """Generates content using the Google GenAI API with retry logic."""
    # Define safety settings using the new types (string format)
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
        ),
    ]
    try:
        # Use client.models.generate_content with GenerateContentConfig
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, safety_settings=safety_settings
            ),
        )
        # Assuming response.text holds the result, similar to the old API
        return response.text
    except errors.APIError as e:
        # Catch specific API errors first
        # Keep existing retry logic trigger for rate limits/resource exhaustion
        if "429" in str(e) or "Resource has been exhausted" in str(e):
            logger.warning(
                f"Rate limit or resource exhaustion (APIError). Retrying... Error: {e}"
            )
            time.sleep(random.uniform(1.0, 5.0))  # Increase sleep slightly
            raise e  # Reraise to trigger tenacity retry
        else:
            # Handle other API errors (log and reraise if not retrying)
            logger.error(f"Caught specific Google GenAI API Error: {e}")
            raise e  # Reraise if not specifically handled for retry
    except Exception as e:
        # Catch any other exceptions (less specific)
        logger.error(f"Caught generic exception during generation: {e}")
        raise e  # Reraise to trigger tenacity retry or fail


def run_gemini_prompt(
    pdf_path: Path, system_prompt: str, prompt: str, model_name: str
) -> str | None:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required.")

    # Initialize the new client
    client = genai.Client(api_key=google_api_key)
    # Upload file using the new client method
    uploaded_file = client.files.upload(file=str(pdf_path))

    try:
        # Pass the client, model_name, system_prompt, file object, and prompt
        result = generate_content_with_retry(
            client, model_name, system_prompt, uploaded_file, prompt
        )
        # Delete the uploaded file after use (optional, good practice)
        client.files.delete(name=uploaded_file.name)
        return result
    except Exception as e:
        logger.error(f"Failed to generate content after multiple retries: {e}")
        # Attempt to delete the file even if generation fails
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception as delete_error:
            logger.error(
                f"Failed to delete uploaded file {uploaded_file.name}: {delete_error}"
            )
        return None
    finally:
        # Ensure file deletion is attempted, wrap in try/except if not already deleted
        try:
            # Check if file still exists (might have been deleted in try or except block)
            client.files.delete(name=uploaded_file.name)
            logger.info(f"Cleaned up uploaded file: {uploaded_file.name}")
        except Exception:
            # Either file doesn't exist or another error occurred during cleanup
            pass


def get_summary(pdf_path: Path, model: str) -> str | None:
    system_prompt = Path("prompts/summary_system_prompt.txt").read_text()
    prompt = Path("prompts/summary_prompt.txt").read_text()
    if "claude" in model.lower():
        logger.info(f"Using Claude model ({model}) for summary for PDF: {pdf_path.name}")
        output = run_claude_prompt(pdf_path, system_prompt, prompt, model, "<summary>")
        if output:
            output = "<summary>\n" + output
    elif "gemini" in model.lower():
        logger.info(f"Using Gemini model ({model}) for summary for PDF: {pdf_path.name}")
        output = run_gemini_prompt(pdf_path, system_prompt, prompt, model)
    else:
        logger.error(f"Unsupported model: {model}")
        return None
    return extract_summary(output) if output else None


def get_tags(pdf_path: Path, model: str) -> List[str] | None:
    system_prompt = Path("prompts/tag_prompt.txt").read_text()
    prompt = Path("prompts/tag_prompt.txt").read_text()
    if "claude" in model.lower():
        logger.info(f"Using Claude model ({model}) for tagging for PDF: {pdf_path.name}")
        output = run_claude_prompt(pdf_path, system_prompt, prompt, model, "<tags>")
        if output:
            output = "<tags>\n" + output
    elif "gemini" in model.lower():
        logger.info(f"Using Gemini model ({model}) for tagging for PDF: {pdf_path.name}")
        output = run_gemini_prompt(pdf_path, system_prompt, prompt, model)
    else:
        logger.error(f"Unsupported model: {model}")
        return None
    logger.info(f"Got tags: {output}")
    return extract_tags(output) if output else None


def summarize_and_tag_single_doc(item, do_summarize: bool = True) -> None:
    key = item["data"]["key"]
    logger.info(f"Handling item {key}")
    title = item["data"].get("title")
    if not title:
        logger.warning(f"Skipping item {key} because it has no title")
        update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
        return

    pdf_items = [
        child
        for child in zot.children(key)
        if child.get("data", {}).get("contentType") == "application/pdf"
    ]
    if not pdf_items:
        logger.error(f"No PDF attachment found for item {key}, skipping.")
        update_item_tags(key, tags_to_add=[ZOTERO_TAGS["DENY"]])
        return

    pdf_path = unzip_pdf(Path(f"zotero/{pdf_items[0]["key"]}.zip"))
    if not pdf_path:
        logger.error(f"Could not find a PDF for item {key} in the path, skipping.")
        update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
        return

    pdf_reader = PdfReader(str(pdf_path))
    if not 5 <= len(pdf_reader.pages) <= 100:
        logger.error("PDF length is out of bounds, skipping.")
        update_item_tags(
            key,
            tags_to_add=[ZOTERO_TAGS["DENY"]],
            tags_to_remove=[ZOTERO_TAGS["TODO"]],
        )
        return

    if do_summarize:
        summary_model = os.getenv("SUMMARY_MODEL", "claude-3-5-sonnet-20240620")
        summary = get_summary(pdf_path, summary_model)
        if not summary:
            logger.error(f"Could not summarize item {key}, skipping.")
            update_item_tags(key, tags_to_add=[ZOTERO_TAGS["ERROR"]])
            return
        write_note(key, f"Summary\n\n{summary}")

    tagging_model = os.getenv("TAG_MODEL", "claude-3-5-sonnet-20240620")
    tags = get_tags(pdf_path, tagging_model)
    update_item_tags(
        key,
        tags_to_add=[ZOTERO_TAGS["SUMMARIZED"]] + tags,
        tags_to_remove=[ZOTERO_TAGS["TODO"]],
    )


def summarize_and_tag_all_docs() -> None:
    items = zot.top(
        tag=[
            ZOTERO_TAGS["TODO"],
            f"-{ZOTERO_TAGS["ERROR"]}",
            f"-{ZOTERO_TAGS["DENY"]}",
        ],
        limit=50,
    )
    logger.info(f"Found {len(items)} items to summarize")
    for item in items:
        summarize_and_tag_single_doc(item, True)


def add_initial_tags() -> None:
    items = zot.top(tag=[f"-{tag}" for tag in ZOTERO_TAGS.values()], limit=50)
    for item in items:
        update_item_tags(item["data"]["key"], tags_to_add=[ZOTERO_TAGS["TODO"]])


def add_missing_tags() -> None:
    items = zot.top(
        tag=[
            ZOTERO_TAGS["SUMMARIZED"],
            f"-{ZOTERO_TAGS["ERROR"]}",
            f"-{ZOTERO_TAGS["DENY"]}",
            f"-{ZOTERO_TAGS["TODO"]}",
        ],
        limit=50,
    )
    items_to_tag = [item for item in items if len(item["data"]["tags"]) < 5]
    logger.info(f"Found {len(items_to_tag)} items to tag")
    for item in items_to_tag:
        summarize_and_tag_single_doc(item, False)


@app.get("/add_initial_tags/")
def fastapi_add_initial_tags():
    try:
        add_initial_tags()
        return {"status": "Tags added successfully!"}
    except Exception as e:
        logger.error(f"Error adding missing tags: {e}")
        return {"status": "Error", "message": str(e)}


@app.get("/add_missing_tags/")
def fastapi_add_missing_tags(background_tasks: BackgroundTasks):
    background_tasks.add_task(add_missing_tags)
    return {"status": "Tags added successfully!"}


@app.get("/summarize/")
def summarize(background_tasks: BackgroundTasks):
    # Add to background tasks as it takes a long time
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

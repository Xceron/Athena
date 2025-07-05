import os
from typing import List

from pyzotero import zotero

# Load configuration from environment variables
ZOTERO_TAGS = {
    "TODO": os.getenv("ZOTERO_TODO_TAG_NAME", "TODO"),
    "SUMMARIZED": os.getenv("ZOTERO_SUMMARIZED_TAG_NAME", "SUMMARIZED"),
    "DENY": os.getenv("ZOTERO_DENY_TAG_NAME", "DENY"),
    "ERROR": os.getenv("ZOTERO_ERROR_TAG_NAME", "ERROR"),
}

# Initialize Zotero client (lazy initialization)
zot = None

def _get_zotero_client():
    """Get or create Zotero client."""
    global zot
    if zot is None:
        api_key = os.getenv("ZOTERO_API_KEY")
        if not api_key:
            raise ValueError("ZOTERO_API_KEY environment variable is required.")
        
        user_id = os.getenv("ZOTERO_USER_ID")
        if not user_id:
            raise ValueError("ZOTERO_USER_ID environment variable is required.")
        
        zot = zotero.Zotero(int(user_id), "user", api_key)
    return zot


def get_todo_items(limit: int = 50) -> List[dict]:
    """Get items that need to be summarized."""
    client = _get_zotero_client()
    return client.top(
        tag=[
            ZOTERO_TAGS["TODO"],
            f"-{ZOTERO_TAGS['ERROR']}",
            f"-{ZOTERO_TAGS['DENY']}",
        ],
        limit=limit,
    )


def get_items_needing_tags(limit: int = 50) -> List[dict]:
    """Get items that need additional tags."""
    client = _get_zotero_client()
    items = client.top(
        tag=[
            ZOTERO_TAGS["SUMMARIZED"],
            f"-{ZOTERO_TAGS['ERROR']}",
            f"-{ZOTERO_TAGS['DENY']}",
            f"-{ZOTERO_TAGS['TODO']}",
        ],
        limit=limit,
    )
    return [item for item in items if len(item["data"]["tags"]) < 5]


def get_items_without_tags(limit: int = 50) -> List[dict]:
    """Get items that have no Athena-related tags."""
    client = _get_zotero_client()
    return client.top(tag=[f"-{tag}" for tag in ZOTERO_TAGS.values()], limit=limit)


def update_item_tags(
    item_id: str,
    tags_to_add: List[str] | None = None,
    tags_to_remove: List[str] | None = None,
) -> None:
    """Update tags for a Zotero item."""
    client = _get_zotero_client()
    item = client.item(item_id)
    current_tags = {tag["tag"] for tag in item["data"]["tags"]}
    
    if tags_to_remove:
        current_tags.difference_update(tags_to_remove)
    if tags_to_add:
        current_tags.update(tags_to_add)
    
    item["data"]["tags"] = [{"tag": tag} for tag in current_tags]
    client.update_item(item)


def write_note(parent_id: str, note_text: str) -> None:
    """Write a note to a Zotero item."""
    client = _get_zotero_client()
    summary_model = os.getenv("SUMMARY_MODEL", "gemini-2.5-pro-exp-03-25")
    template = {
        "itemType": "note",
        "parentItem": parent_id,
        "tags": [{"tag": summary_model}, {"tag": ZOTERO_TAGS["SUMMARIZED"]}],
        "note": note_text.replace("\n", "<br>"),
    }
    client.create_items([template])


def get_pdf_children(item_key: str) -> List[dict]:
    """Get PDF attachments for a Zotero item."""
    client = _get_zotero_client()
    return [
        child
        for child in client.children(item_key)
        if child.get("data", {}).get("contentType") == "application/pdf"
    ]

import os
import random
import re
import time
from pathlib import Path
from typing import List

from anthropic import Anthropic
from google import genai
from google.genai import types, errors
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pdf_utils import extract_text_from_pdf


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent
    return (project_root / "prompts" / f"{name}.txt").read_text().strip()


def extract_summary(text: str) -> str | None:
    """Extract summary from LLM response."""
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_tags(text: str) -> List[str]:
    """Extract tags from LLM response."""
    match = re.search(r"<tags>(.*?)</tags>", text, re.DOTALL)
    if not match:
        return []
    
    tag_text = match.group(1).strip()
    raw_tags = re.split(r"[,\n]+", tag_text)
    cleaned_tags = []
    
    for tag in raw_tags:
        tag = tag.strip()
        if not tag:
            continue
        # Remove leading list markers
        tag = re.sub(r"^[\\s•-]+", "", tag)
        # Remove parenthetical explanations
        tag = re.sub(r"\\s*\\([^)]*\\)", "", tag)
        tag = tag.strip()
        if tag:
            cleaned_tags.append(tag)
    
    return cleaned_tags


class LLMRouter:
    """Unified interface for LLM operations.
    
    Cache behavior:
    - summary_and_tags(): Uses summary system prompt for both operations to enable cache hits
    - tags_only(): Uses dedicated tag system prompt for better accuracy (no cache benefits)
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_claude = "claude" in model_name.lower()
        self.is_gemini = "gemini" in model_name.lower()
        
        # Cache prompts - Use summary system prompt for both operations to ensure cache hits
        self.summary_system_prompt = load_prompt("summary_system_prompt")
        self.summary_prompt = load_prompt("summary_prompt")
        self.tag_prompt = load_prompt("tag_prompt")
        
        # Load separate tag system prompt for tags-only operations
        self.tag_system_prompt = load_prompt("tag_system_prompt")
        
        # Initialize clients
        if self.is_claude:
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key:
                raise ValueError("CLAUDE_API_KEY environment variable is required.")
            self.client = Anthropic(api_key=api_key)
        elif self.is_gemini:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required.")
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def summary_and_tags(self, pdf_path: Path) -> tuple[str | None, List[str] | None]:
        """Get both summary and tags in a single conversation."""
        if self.is_claude:
            return self._run_claude_summary_and_tags(pdf_path)
        elif self.is_gemini:
            return self._run_gemini_summary_and_tags(pdf_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def tags_only(self, pdf_path: Path) -> List[str] | None:
        """Get tags only for an already summarized document."""
        if self.is_claude:
            return self._run_claude_tags_only(pdf_path)
        elif self.is_gemini:
            return self._run_gemini_tags_only(pdf_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _run_claude_summary_and_tags(self, pdf_path: Path) -> tuple[str | None, List[str] | None]:
        """Run Claude conversation: PDF -> summary -> tags."""
        with pdf_path.open("rb") as pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file.read())
        if not pdf_text:
            return None, None
        
        summary_user_prompt = f"{self.summary_prompt}\n\n<paper>\n{pdf_text}\n</paper>"
        
        try:
            # Step 1: Get summary
            summary_message = self.client.messages.create(
                system=self.summary_system_prompt,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": summary_user_prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "<summary>"}]},
                ],
                model=self.model_name,
            )
            summary_response = "<summary>\n" + summary_message.content[0].text
            summary = extract_summary(summary_response)
            
            if not summary:
                return None, None
            
            # Step 2: Get tags using the summary
            tag_message = self.client.messages.create(
                system=self.summary_system_prompt,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": summary_user_prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": summary_response}]},
                    {"role": "user", "content": [{"type": "text", "text": self.tag_prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "<tags>"}]},
                ],
                model=self.model_name,
            )
            tag_response = "<tags>\n" + tag_message.content[0].text
            tags = extract_tags(tag_response)
            
            return summary, tags
            
        except Exception:
            return None, None
    
    def _run_claude_tags_only(self, pdf_path: Path) -> List[str] | None:
        """Run Claude for tags only - uses separate tag system prompt."""
        with pdf_path.open("rb") as pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file.read())
        if not pdf_text:
            return None
        
        user_prompt = f"{self.tag_prompt}\n\n<paper>\n{pdf_text}\n</paper>"
        
        try:
            message = self.client.messages.create(
                system=self.tag_system_prompt,  # Use proper tag system prompt
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "<tags>"}]},
                ],
                model=self.model_name,
            )
            tag_response = "<tags>\n" + message.content[0].text
            return extract_tags(tag_response)
        except Exception:
            return None
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _generate_content_with_retry(self, uploaded_file: types.File, prompt: str):
        """Generate content using Gemini API with retry logic."""
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, uploaded_file],
                config=types.GenerateContentConfig(
                    system_instruction=self.summary_system_prompt,
                    safety_settings=safety_settings
                ),
            )
            return response.text
        except errors.APIError as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                time.sleep(random.uniform(1.0, 5.0))
                raise e
            else:
                raise e
    
    def _run_gemini_summary_and_tags(self, pdf_path: Path) -> tuple[str | None, List[str] | None]:
        """Run Gemini conversation: PDF -> summary -> tags."""
        uploaded_file = self.client.files.upload(file=str(pdf_path))
        
        try:
            # Step 1: Get summary
            summary_result = self._generate_content_with_retry(uploaded_file, self.summary_prompt)
            if not summary_result:
                return None, None
            
            summary = extract_summary(summary_result)
            if not summary:
                return None, None
            
            # Step 2: Get tags using the summary
            tag_result = self._generate_content_with_retry(
                uploaded_file, 
                f"{self.summary_prompt}\n\nSummary: {summary}\n\n{self.tag_prompt}"
            )
            if not tag_result:
                return summary, None
            
            tags = extract_tags(tag_result)
            return summary, tags
            
        except Exception:
            return None, None
        finally:
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
    
    def _run_gemini_tags_only(self, pdf_path: Path) -> List[str] | None:
        """Run Gemini for tags only - uses separate tag system prompt."""
        uploaded_file = self.client.files.upload(file=str(pdf_path))
        
        try:
            # Create a version of generate_content_with_retry that uses tag system prompt
            safety_settings = [
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.tag_prompt, uploaded_file],
                config=types.GenerateContentConfig(
                    system_instruction=self.tag_system_prompt,  # Use proper tag system prompt
                    safety_settings=safety_settings
                ),
            )
            
            if not response.text:
                return None
            
            return extract_tags(response.text)
            
        except Exception:
            return None
        finally:
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

"""
Gemini API Resilience and Retry Utilities.

Provides exponential backoff retry logic and transparent fallback model support
to gracefully handle transient Gemini API capacity spikes (503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED).
"""

import time
import logging
from google.genai.errors import APIError

logger = logging.getLogger("gemini_utils")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_KEYWORDS = [
    "503",
    "unavailable",
    "high demand",
    "resource_exhausted",
    "rate limit",
    "quota",
    "overloaded",
    "connection reset",
    "temporarily unavailable",
    "try again later"
]


def is_retryable_error(error: Exception) -> bool:
    """Check if an exception represents a transient/retryable Gemini API condition."""
    if isinstance(error, APIError):
        code = getattr(error, "code", None)
        if code in RETRYABLE_STATUS_CODES:
            return True

    msg = str(error).lower()
    return any(keyword in msg for keyword in RETRYABLE_KEYWORDS)


def format_gemini_error(error: Exception) -> dict:
    """
    Format a Gemini error into a human-friendly message and category for UI display.

    Returns:
        dict: {"category": str, "title": str, "message": str, "raw": str}
    """
    msg = str(error)
    lower = msg.lower()

    if "503" in lower or "unavailable" in lower or "high demand" in lower:
        return {
            "category": "high_demand",
            "title": "Gemini Model Capacity Spike",
            "message": (
                "Google Gemini servers are currently experiencing high demand spikes. "
                "Automatic retries were attempted. Please wait 5–10 seconds and try again, "
                "or provide your own Gemini API key in the sidebar for dedicated quota."
            ),
            "raw": msg,
        }

    if "429" in lower or "resource_exhausted" in lower or "quota" in lower or "rate limit" in lower:
        return {
            "category": "rate_limit",
            "title": "API Rate Limit Exceeded",
            "message": (
                "Gemini API rate limit or quota exceeded for the current API key. "
                "Please wait 30–60 seconds, or enter your own free Gemini API key in the sidebar under 'API Configuration' "
                "(get a free key at aistudio.google.com)."
            ),
            "raw": msg,
        }

    if any(k in lower for k in ["api_key_invalid", "api key not valid", "invalid api key"]) or (
        "400" in lower and "api_key" in lower
    ):
        return {
            "category": "auth",
            "title": "Invalid Gemini API Key",
            "message": (
                "The configured Gemini API key appears invalid or expired. "
                "Please verify your key in the sidebar under 'API Configuration' or at Google AI Studio."
            ),
            "raw": msg,
        }

    return {
        "category": "general",
        "title": "Recommendation Request Failed",
        "message": (
            "An error occurred while generating product recommendations. "
            "Please check the technical details below and try again."
        ),
        "raw": msg,
    }


def generate_content_with_retry(
    client,
    model: str,
    contents,
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    fallback_models: list = None,
    **kwargs
):
    """
    Execute client.models.generate_content with exponential backoff and fallback models.

    Args:
        client: google.genai.Client instance
        model: Primary Gemini model name (e.g. 'gemini-2.5-flash')
        contents: Text/image contents for generation
        max_retries: Maximum attempts on the primary model
        initial_delay: Initial wait duration in seconds before first retry
        backoff_factor: Multiplier for wait duration on subsequent retries
        fallback_models: Optional list of fallback model names to try if primary fails
        **kwargs: Additional parameters passed to generate_content
    """
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                **kwargs
            )
        except Exception as e:
            last_error = e
            if is_retryable_error(e) and attempt < max_retries:
                logger.warning(
                    f"[Gemini Retry] Attempt {attempt}/{max_retries} failed on model '{model}': {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                break

    # If primary model exhausted retries and fallback models provided, try them
    if fallback_models:
        for fallback_model in fallback_models:
            if fallback_model == model:
                continue
            logger.warning(
                f"[Gemini Fallback] Primary model '{model}' unavailable. Attempting fallback model '{fallback_model}'..."
            )
            try:
                result = client.models.generate_content(
                    model=fallback_model,
                    contents=contents,
                    **kwargs
                )
                logger.info(f"[Gemini Fallback] Succeeded with fallback model '{fallback_model}'.")
                return result
            except Exception as fb_err:
                logger.warning(f"[Gemini Fallback] Fallback model '{fallback_model}' also failed: {fb_err}")
                last_error = fb_err

    if last_error:
        raise last_error


def embed_content_with_retry(
    client,
    model: str,
    contents,
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs
):
    """
    Execute client.models.embed_content with exponential backoff for transient errors.
    """
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.embed_content(
                model=model,
                contents=contents,
                **kwargs
            )
        except Exception as e:
            last_error = e
            if is_retryable_error(e) and attempt < max_retries:
                logger.warning(
                    f"[Gemini Embedding Retry] Attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                break

    if last_error:
        raise last_error

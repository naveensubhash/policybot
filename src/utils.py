import hashlib
import uuid
from datetime import datetime


def hash_text(text: str) -> str:
    """Generate SHA-256 hash of text for reproducibility tracking"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_run_id() -> str:
    """Generate unique run identifier"""
    return str(uuid.uuid4())


def current_timestamp() -> str:
    """Get current UTC timestamp in ISO format"""
    return datetime.utcnow().isoformat()
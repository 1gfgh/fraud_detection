import hashlib


def get_string_hash(s) -> str:
    raw_bytes = s.encode('utf-8')
    truncated = raw_bytes[:1000]
    return hashlib.sha256(truncated).hexdigest()[:32]

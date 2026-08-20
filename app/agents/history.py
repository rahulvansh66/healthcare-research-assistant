def format_history(messages: list[dict], max_chars: int) -> str:
    """Render prior turns as 'Role: content' lines, keeping the most recent
    turns that fit within max_chars — oldest turns are dropped first so
    prompt size stays bounded regardless of conversation length."""
    lines = [
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in messages
    ]

    kept = []
    total = 0
    for line in reversed(lines):
        total += len(line) + 1
        if total > max_chars and kept:
            break
        kept.append(line)

    if not kept:
        return ""
    return "\n".join(reversed(kept)) + "\n"

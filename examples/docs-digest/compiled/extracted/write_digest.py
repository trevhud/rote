def write_digest(request: dict, pick_reading_list: dict, read_repo_outline: str) -> dict:
    """Build the markdown digest."""
    if not isinstance(pick_reading_list, dict) or "reading_list" not in pick_reading_list:
        raise ValueError("pick_reading_list must contain 'reading_list'")
    items = pick_reading_list["reading_list"]
    if not isinstance(items, list):
        raise ValueError("reading_list must be a list")
    if not isinstance(read_repo_outline, str):
        raise ValueError("read_repo_outline must be a string")
    lines = ["## Reading list", ""]
    for item in items:
        if not isinstance(item, dict) or "title" not in item or "url" not in item:
            raise ValueError("each reading list item needs title and url")
        # skill: "one bullet per kept result, the title linked to its URL."
        lines.append("- [" + item["title"] + "](" + item["url"] + ")")
    lines.append("")
    lines.append("## Repo outline")
    lines.append("")
    # skill: "the wiki outline, as the server returned it."
    lines.append(read_repo_outline)
    return {"digest": "\n".join(lines) + "\n"}

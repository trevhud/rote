"""Build the markdown digest."""

# skill: "Return a markdown document with two sections:"
FIRST_SECTION = "Reading list"
SECOND_SECTION = "Repo outline"


def write_digest(request: dict, pick_reading_list: dict, read_repo_wiki_outline: str, first_section: str = 'Reading list', second_section: str = 'Repo outline') -> dict:
    if not isinstance(pick_reading_list, dict) or "reading_list" not in pick_reading_list:
        raise ValueError("missing reading_list")
    items = pick_reading_list["reading_list"]
    if not isinstance(items, list):
        raise ValueError("reading_list must be a list")
    if not isinstance(read_repo_wiki_outline, str):
        raise ValueError("wiki outline must be a string")
    lines = ["## " + first_section, ""]
    for item in items:
        if not isinstance(item, dict) or "title" not in item or "url" not in item:
            raise ValueError("each item needs title and url")
        # skill: "one bullet per kept result, the title linked to its URL."
        lines.append("- [" + str(item["title"]) + "](" + str(item["url"]) + ")")
    lines.append("")
    lines.append("## " + second_section)
    lines.append("")
    # skill: "the wiki outline, as the server returned it."
    lines.append(read_repo_wiki_outline)
    return {"digest": "\n".join(lines) + "\n"}

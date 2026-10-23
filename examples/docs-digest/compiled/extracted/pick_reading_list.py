RESULT_LIMIT = 5  # skill: "Keep the first five results."


def pick_reading_list(request: dict, search_microsoft_learn: dict, result_limit: str = 'first five') -> dict:
    """Keep the first five search results, each reduced to title and url."""
    if not isinstance(search_microsoft_learn, dict):
        raise ValueError("search_microsoft_learn must be an object with a 'results' list")
    results = search_microsoft_learn.get("results")
    if not isinstance(results, list):
        raise ValueError("search_microsoft_learn is missing 'results' list")
    reading_list = []
    # skill: "Keep the first five results. For each, keep its title and its link."
    for item in results[:RESULT_LIMIT]:
        if not isinstance(item, dict):
            raise ValueError("each result must be an object")
        reading_list.append({
            "title": item.get("title") or "",
            "url": item.get("contentUrl") or "",
        })
    return {"reading_list": reading_list}

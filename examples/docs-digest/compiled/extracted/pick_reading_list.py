def pick_reading_list(request: dict, search_microsoft_learn: dict, result_count: int = 5) -> dict:
    """Keep the first results, each reduced to title and url."""
    if not isinstance(search_microsoft_learn, dict):
        raise ValueError("search_microsoft_learn must be an object with a 'results' array")
    results = search_microsoft_learn.get("results")
    if results is None:
        results = []
    if not isinstance(results, list):
        raise ValueError("'results' must be an array")
    picked = []
    # skill: "Keep the first 5 results. For each, keep its title and its link."
    for item in results[:result_count]:
        if not isinstance(item, dict):
            raise ValueError("each result must be an object")
        picked.append(
            {
                "title": item.get("title") or "",
                "url": item.get("contentUrl") or "",
            }
        )
    return {"reading_list": picked}

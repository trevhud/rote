import httpx


def f(request):
    pulls = httpx.get(
        f"https://api.github.com/repos/{request['owner']}/{request['repo']}/pulls",
        params={"state": "all", "sort": "updated", "direction": "desc", "per_page": 50},
    ).json()
    return [
        {"branch": p["head"]["ref"], "merged": p["merged_at"], "author": p["user"]["login"]}
        for p in pulls
    ]

import httpx

GITHUB = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def f(request):
    owner, repo = request["owner"], request["repo"]
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        issues = client.get(
            f"{GITHUB}/repos/{owner}/{repo}/issues",
            params={"state": "open", "labels": "bug,triage", "per_page": 100, "sort": "updated"},
        ).json()
        opened = client.post(
            f"{GITHUB}/repos/{owner}/{repo}/issues",
            json={"title": request["title"], "body": request["body"], "labels": ["bug"]},
        ).json()
        comment = client.post(
            f"{GITHUB}/repos/{owner}/{repo}/issues/{opened['number']}/comments",
            json={"body": "Triaged by the pipeline."},
        ).json()
        found = client.get(
            f"{GITHUB}/search/issues", params={"q": f"repo:{owner}/{repo} is:issue is:open"}
        ).json()
    return {
        "open": [
            {"n": i["number"], "title": i["title"], "pr": "pull_request" in i} for i in issues
        ],
        "url": opened["html_url"],
        "comment": comment["id"],
        "total": found["total_count"],
        "hits": [h["html_url"] for h in found["items"]],
    }

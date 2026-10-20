import base64

import requests

API = "https://api.github.com"


def f(request):
    owner, repo = request["owner"], request["repo"]
    meta = requests.get(f"{API}/repos/{owner}/{repo}", timeout=30).json()
    readme = requests.get(
        f"{API}/repos/{owner}/{repo}/contents/docs/README.md", params={"ref": "main"}, timeout=30
    ).json()
    ref = requests.get(f"{API}/repos/{owner}/{repo}/git/ref/heads/main", timeout=30).json()
    commits = requests.get(
        f"{API}/repos/{owner}/{repo}/commits",
        params={"sha": "main", "since": request["since"], "per_page": 20},
        timeout=30,
    ).json()
    user = requests.get(f"{API}/users/{meta['owner']['login']}", timeout=30).json()
    return {
        "stars": meta["stargazers_count"],
        "branch": meta["default_branch"],
        "readme": base64.b64decode(readme["content"]).decode(),
        "sha": readme["sha"],
        "head": ref["object"]["sha"],
        "log": [(c["sha"], c["commit"]["message"], c["commit"]["author"]["date"]) for c in commits],
        "login": user["login"],
        "public_repos": user["public_repos"],
    }

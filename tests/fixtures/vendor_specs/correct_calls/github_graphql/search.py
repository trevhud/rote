import os

import requests

SEARCH = """
query Search($q: String!) {
  search(query: $q, type: ISSUE, first: 20) {
    issueCount
    nodes {
      ... on Issue { title url state }
      ... on PullRequest { title url mergedAt }
    }
  }
}

query Viewer { viewer { login } }
"""


def f(request):
    resp = requests.post(
        "https://api.github.com/graphql",
        json={"query": SEARCH, "operationName": "Search", "variables": {"q": request["q"]}},
        headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"]},
        timeout=30,
    )
    data = resp.json()["data"]
    return {
        "count": data["search"]["issueCount"],
        "hits": [(n["title"], n["url"]) for n in data["search"]["nodes"]],
    }

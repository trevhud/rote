import os

import httpx

ENDPOINT = "https://api.github.com/graphql"

ISSUES = """
query ($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    issues(first: 50, after: $cursor, states: OPEN, orderBy: {field: UPDATED_AT, direction: DESC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        url
        author { login }
        labels(first: 10) { nodes { name color } }
      }
    }
  }
}
"""


def f(request):
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
    variables = {"owner": request["owner"], "name": request["repo"], "cursor": None}
    body = httpx.post(
        ENDPOINT, json={"query": ISSUES, "variables": variables}, headers=headers
    ).json()
    if body.get("errors"):
        raise RuntimeError(body["errors"][0]["message"])
    issues = body["data"]["repository"]["issues"]
    return {
        "repo": body["data"]["repository"]["nameWithOwner"],
        "total": issues["totalCount"],
        "next": issues["pageInfo"]["endCursor"] if issues["pageInfo"]["hasNextPage"] else None,
        "issues": [
            {
                "n": i["number"],
                "title": i["title"],
                "url": i["url"],
                "by": i["author"]["login"],
                "labels": [label["name"] for label in i["labels"]["nodes"]],
            }
            for i in issues["nodes"]
        ],
    }

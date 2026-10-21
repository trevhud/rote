import os

import httpx

QUERY = """
fragment Repo on Repository {
  nameWithOwner
  stargazerCount
  primaryLanguage { name }
  defaultBranchRef { target { ... on Commit { oid committedDate } } }
}

query ($login: String!) {
  user(login: $login) {
    login
    repositories(
      first: 10, ownerAffiliations: OWNER, orderBy: {field: STARGAZERS, direction: DESC}
    ) {
      nodes { ...Repo }
    }
  }
}
"""


def f(request):
    body = httpx.post(
        "https://api.github.com/graphql",
        json={"query": QUERY, "variables": {"login": request["login"]}},
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
    ).json()
    repos = body["data"]["user"]["repositories"]["nodes"]
    return [
        {
            "repo": r["nameWithOwner"],
            "stars": r["stargazerCount"],
            "language": (r["primaryLanguage"] or {}).get("name"),
            "head": r["defaultBranchRef"]["target"]["oid"],
        }
        for r in repos
    ]

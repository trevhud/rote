import os

import httpx

ADD_COMMENT = """
mutation ($subject: ID!, $body: String!) {
  addComment(input: {subjectId: $subject, body: $body}) {
    commentEdge { node { url } }
  }
}
"""


def f(request):
    body = httpx.post(
        "https://api.github.com/graphql",
        json={
            "query": ADD_COMMENT,
            "variables": {"subject": request["issue_id"], "body": request["text"]},
        },
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
    ).json()
    return body["data"]["addComment"]["commentEdge"]["node"]["url"]

import requests

QUERY = """
query ($user: String!) {
  MediaListCollection(userName: $user, type: ANIME, status: COMPLETED) {
    lists {
      name
      entries {
        score
        advancedScores
        media { id title { romaji } }
      }
    }
  }
}
"""


def f(request):
    data = requests.post(
        "https://graphql.anilist.co",
        json={"query": QUERY, "variables": {"user": request["user"]}},
        timeout=30,
    ).json()["data"]
    out = []
    for group in data["MediaListCollection"]["lists"]:
        for entry in group["entries"]:
            out.append(
                {
                    "title": entry["media"]["title"]["romaji"],
                    "score": entry["score"],
                    "story": entry["advancedScores"]["Story"],
                }
            )
    return out

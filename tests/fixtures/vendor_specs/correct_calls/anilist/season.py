import httpx

QUERY = """
query ($season: MediaSeason, $year: Int, $page: Int) {
  Page(page: $page, perPage: 25) {
    pageInfo { hasNextPage currentPage }
    media(season: $season, seasonYear: $year, type: ANIME, sort: POPULARITY_DESC) {
      id
      title { romaji english }
      averageScore
      episodes
      genres
      nextAiringEpisode { airingAt episode }
    }
  }
}
"""


def f(request):
    variables = {"season": "WINTER", "year": 2026, "page": 1}
    body = httpx.post(
        "https://graphql.anilist.co", json={"query": QUERY, "variables": variables}
    ).json()
    page = body["data"]["Page"]
    return {
        "more": page["pageInfo"]["hasNextPage"],
        "rows": [
            {
                "id": m["id"],
                "title": m["title"]["english"] or m["title"]["romaji"],
                "score": m.get("averageScore"),
                "next": (m["nextAiringEpisode"] or {}).get("episode"),
            }
            for m in page["media"]
        ],
    }

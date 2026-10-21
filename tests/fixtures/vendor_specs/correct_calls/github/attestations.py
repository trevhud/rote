import httpx


def f(request):
    r = httpx.get(
        f"https://api.github.com/orgs/{request['org']}/attestations/{request['digest']}",
        params={"per_page": 10},
    )
    return [a["bundle"]["verificationMaterial"]["tlogEntries"] for a in r.json()["attestations"]]

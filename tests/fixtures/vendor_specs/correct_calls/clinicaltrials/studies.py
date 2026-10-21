import httpx

URL = "https://clinicaltrials.gov/api/v2/studies"


def f(request):
    params = {
        "query.cond": request["condition"],
        "query.term": request.get("term", ""),
        "filter.overallStatus": "RECRUITING|NOT_YET_RECRUITING",
        "fields": "NCTId,BriefTitle,OverallStatus",
        "sort": ["LastUpdatePostDate:desc"],
        "pageSize": 20,
        "countTotal": "true",
        "format": "json",
    }
    data = httpx.get(URL, params=params, timeout=30).json()
    out = []
    for s in data["studies"]:
        ps = s["protocolSection"]
        out.append(
            {
                "id": ps["identificationModule"]["nctId"],
                "title": ps["identificationModule"]["briefTitle"],
                "status": ps["statusModule"]["overallStatus"],
            }
        )
    nxt = httpx.get(URL, params={**params, "pageToken": data["nextPageToken"]}, timeout=30).json()
    one = httpx.get(f"https://clinicaltrials.gov/api/v2/studies/{out[0]['id']}").json()
    return {
        "studies": out,
        "total": data.get("totalCount"),
        "more": len(nxt["studies"]),
        "has_results": one["hasResults"],
    }

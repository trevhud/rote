import httpx


def f(request):
    page = httpx.get(
        "https://api.stripe.com/v1/subscriptions?limit=3&expand[]=data.default_payment_method",
        params={"status": "active", "customer": request["customer"]},
        auth=(request["key"], ""),
    ).json()
    prices = []
    for sub in page["data"]:
        for item in sub["items"]["data"]:
            prices.append(item["price"]["id"])
    return {"prices": prices, "status": [s["status"] for s in page["data"]]}

import requests


def f(request):
    resp = requests.get(
        "https://api.stripe.com/v1/charges",
        params={
            "created[gte]": request["since"],
            "created[lte]": request["until"],
            "limit": 100,
            "customer": request["customer"],
        },
        auth=(request["key"], ""),
        timeout=30,
    )
    page = resp.json()
    out = []
    for charge in page["data"]:
        out.append(
            {
                "id": charge["id"],
                "amount": charge["amount"],
                "paid": charge["paid"],
                "customer": charge["customer"],
                "card": charge["payment_method_details"]["card"]["brand"],
            }
        )
    return out

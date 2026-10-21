import httpx

API = "https://api.stripe.com/v1"


def f(request):
    auth = (request["key"], "")
    intent = httpx.post(
        f"{API}/payment_intents",
        data={
            "amount": request["amount"],
            "currency": "usd",
            "customer": request["customer"],
            "automatic_payment_methods[enabled]": "true",
            "metadata[order_id]": request["order"],
        },
        auth=auth,
    ).json()
    again = httpx.get(f"{API}/payment_intents/{intent['id']}", auth=auth).json()
    refund = httpx.post(
        f"{API}/refunds",
        data={"payment_intent": intent["id"], "reason": "requested_by_customer"},
        auth=auth,
    ).json()
    session = httpx.post(
        f"{API}/checkout/sessions",
        data={
            "mode": "payment",
            "success_url": request["success_url"],
            "line_items[0][price]": request["price"],
            "line_items[0][quantity]": 1,
        },
        auth=auth,
    ).json()
    return {
        "secret": intent["client_secret"],
        "status": again["status"],
        "charge": again["latest_charge"],
        "refund": refund["status"],
        "url": session["url"],
    }

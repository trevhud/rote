import httpx

API = "https://api.stripe.com/v1"


def f(request):
    auth = (request["key"], "")
    listed = httpx.get(
        f"{API}/customers",
        params={"email": request["email"], "limit": 10, "expand[]": "data.default_source"},
        auth=auth,
    ).json()
    ids = [c["id"] for c in listed["data"]]
    created = httpx.post(
        f"{API}/customers",
        data={
            "email": request["email"],
            "name": request["name"],
            "metadata[user_id]": request["user_id"],
            "address[city]": request["city"],
            "preferred_locales[0]": "en",
        },
        auth=auth,
    ).json()
    one = httpx.get(
        f"{API}/customers/{created['id']}",
        params={"expand[]": ["default_source", "invoice_settings.default_payment_method"]},
        auth=auth,
    ).json()
    found = httpx.get(
        f"{API}/customers/search",
        params={"query": f"email:'{request['email']}'", "limit": 5},
        auth=auth,
    ).json()
    gone = httpx.delete(f"{API}/customers/{ids[0]}", auth=auth).json()
    return {
        "ids": ids,
        "more": listed["has_more"],
        "email": one.get("email"),
        "deleted": gone["deleted"],
        "next": found.get("next_page"),
        "found": [c["email"] for c in found["data"]],
    }

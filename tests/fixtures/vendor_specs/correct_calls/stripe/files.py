import httpx


def f(request):
    uploaded = httpx.post(
        "https://files.stripe.com/v1/files",
        data={"purpose": "dispute_evidence"},
        files={"file": request["file"]},
        auth=(request["key"], ""),
    ).json()
    return {"id": uploaded["id"], "size": uploaded["size"]}

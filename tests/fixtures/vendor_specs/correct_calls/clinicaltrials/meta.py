import requests

BASE = "https://clinicaltrials.gov/api/v2"


def f(request):
    version = requests.get(BASE + "/version", timeout=30).json()
    size = requests.get(BASE + "/stats/size", timeout=30).json()
    return {
        "api": version["apiVersion"],
        "data": version["dataTimestamp"],
        "average": size["averageSizeBytes"],
    }

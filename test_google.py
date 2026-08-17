# test_google.py — debug Google Custom Search

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

k = os.getenv("GOOGLE_API_KEY")
cx = os.getenv("GOOGLE_CX")

print("GOOGLE_API_KEY present:", bool(k))
print("GOOGLE_CX present:", bool(cx))

q = "open to work software engineer -linkedin -indeed -glassdoor"

try:
    r = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": k,
            "cx": cx,
            "q": q,
            "num": 3,
        },
        timeout=15,
    )

    print("Status:", r.status_code)

    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text[:2000])

except Exception as e:
    print("Network error:", e)

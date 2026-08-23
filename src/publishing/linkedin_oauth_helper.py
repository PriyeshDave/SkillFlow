"""
One-time, manual helper to obtain a LinkedIn access token. You can also
just reuse the access token and person URN already generated for BlogMind
(the same LinkedIn Developer App works fine across projects) -- run this
only if you want a fresh token or a separate app.

Usage:
    python -m src.publishing.linkedin_oauth_helper
"""
from __future__ import annotations

import os
import urllib.parse
import webbrowser

import requests

AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
REDIRECT_URI = "http://localhost:8765/callback"
SCOPES = "openid profile w_member_social"


def get_authorization_url() -> str:
    client_id = os.environ["LINKEDIN_CLIENT_ID"]
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    client_id = os.environ["LINKEDIN_CLIENT_ID"]
    client_secret = os.environ["LINKEDIN_CLIENT_SECRET"]

    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_person_urn(access_token: str) -> str:
    resp = requests.get(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    sub = resp.json()["sub"]
    return f"urn:li:person:{sub}"


if __name__ == "__main__":
    url = get_authorization_url()
    print(f"1. Open this URL and authorize the app:\n{url}\n")
    webbrowser.open(url)
    print("2. After redirect, copy the 'code' query param from the URL bar.")
    code = input("Paste the code here: ").strip()

    token_data = exchange_code_for_token(code)
    access_token = token_data["access_token"]
    print(f"\nAccess token (expires in {token_data.get('expires_in')}s):\n{access_token}")

    urn = get_person_urn(access_token)
    print(f"\nYour person URN:\n{urn}")
    print(
        "\nStore these as GitHub Actions secrets: "
        "LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN."
    )

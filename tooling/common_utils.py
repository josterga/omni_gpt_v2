# tooling/common_utils.py
from bs4 import BeautifulSoup

SLACK_EXCLUSIONS = (
    " -in:customer-sla-breach -in:customer-triage -in:support-overflow "
    "-in:omnis -in:customer-membership-alerts -in:vector-alerts -in:notifications-alerts "
    "-cypress -github -sentry -squadcast -syften -in:leadership -in:leaders"
)

def make_docs_url_from_path(raw_path: str) -> str:
    # ./docs/10-guides/getting-started.md -> https://docs.omni.co/guides/getting-started
    base = "https://docs.omni.co/"
    trimmed = raw_path.replace("./docs/", "", 1).removesuffix(".md")
    parts = trimmed.split("/", 1)
    if "-" in parts[0] and parts[0].split("-")[0].isdigit():
        parts[0] = "-".join(parts[0].split("-")[1:])
    return base + "/".join(parts)

def make_community_url(slug: str, topic_id: str) -> str | None:
    if slug and topic_id:
        return f"https://community.omni.co/t/{slug}/{topic_id}/"
    return None

def html_to_text(s: str) -> str:
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

def dedupe_by_url_or_text(items):
    seen, out = set(), []
    for it in items:
        key = (it.get("url") or "") + "|" + (it.get("text") or it.get("content",""))
        if key in seen: 
            continue
        seen.add(key)
        out.append(it)
    return out

# evidence.py
from typing import Any, Dict, List
import json as _json

def flatten_for_synth(
    evidence: List[Dict[str, Any]],
    *,
    mode: str = "direct",     # "planned" or "direct"
    json_len: int = None,
    text_len: int = None,
    max_docs: int = None
) -> List[Dict[str, str]]:
    # Mode-based defaults
    if mode == "planned":
        json_len = json_len or 1600
        text_len = text_len or 4000
        max_docs = max_docs or 100
    else:  # direct
        json_len = json_len or 800
        text_len = text_len or 1200
        max_docs = max_docs or 30

    docs: List[Dict[str, str]] = []
    for item in evidence or []:
        kind = item.get("kind")
        src = item.get("source") or item.get("tool") or "tool"
        val = item.get("value")

        # Skip completely empty values except errors
        if not val and kind != "error":
            continue

        if kind == "docs" and isinstance(val, list):
            for d in val:
                docs.append({
                    "title": d.get("title") or d.get("source") or "doc",
                    "url": d.get("url") or "",
                    "content": d.get("text") or "",
                    "source": d.get("source") or src,
                })

        elif kind == "text" and isinstance(val, str):
            docs.append({
                "title": src, "url": "", "content": val[:text_len], "source": src
            })

        elif kind == "json":
            if src == "fathom_list_meetings":
                try:
                    meetings = val if isinstance(val, list) else val.get("meetings", [])
                    summaries = []
                    for m in meetings[:10]:  # cap at 10 meetings
                        title = m.get("meeting_title") or m.get("title") or "Untitled"
                        mtype = (m.get("meeting_type") or "").capitalize()
                        created_at = m.get("created_at")
                        url = m.get("url") or ""
                        share_url = m.get("share_url", "")

                        summary_text = ""
                        summary_data = m.get("default_summary")
                        if isinstance(summary_data, dict):
                            summary_text = summary_data.get("markdown_formatted") or summary_data.get("text") or ""
                        if summary_text:
                            summary_text = summary_text.strip().replace("\n", " ")[:700] + "..."  # trim

                        actions_data = m.get("action_items") or []
                        action_texts = []
                        for a in actions_data[:3]:  # cap at 3 actions
                            if isinstance(a, dict) and a.get("description"):
                                action_texts.append(a["description"])

                        bullet = (
                            f"- **{title}** ({mtype}, {created_at})\n"
                            f"  URL: {url} | Share: {share_url}\n"
                            f"  _Summary_: {summary_text or 'No summary'}"
                        )
                        if action_texts:
                            bullet += "\n  _Action Items_: " + "; ".join(action_texts)

                        summaries.append(bullet)

                    compact = "\n\n".join(summaries)
                except Exception:
                    compact = _json.dumps(val, ensure_ascii=False)[:json_len]

                docs.append({
                    "title": f"{src} (json)",
                    "url": "",
                    "content": compact,
                    "source": src
                })

            else:
                docs.append({
                    "title": f"{src} (json)",
                    "url": "",
                    "content": _json.dumps(val, ensure_ascii=False)[:json_len],
                    "source": src
                })

        elif kind == "error":
            docs.append({
                "title": f"{src} (error)", "url": "", "content": str(val), "source": src
            })

        else:
            docs.append({
                "title": f"{src} ({kind or 'unknown'})",
                "url": "",
                "content": str(val)[:text_len],
                "source": src
            })

    return docs[:max_docs]

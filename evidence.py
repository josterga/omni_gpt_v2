# evidence.py
from typing import Any, Dict, List
import json as _json

def flatten_for_synth(
    evidence: List[Dict[str, Any]],
    *,
    json_len: int = 800,
    text_len: int = 1200,
    max_docs: int = 30
) -> List[Dict[str, str]]:
    """
    Accepts mixed evidence items:
      {"kind":"docs","value":[{text,source,url, title?},...]}
      {"kind":"text","value":"..."}
      {"kind":"json","value":[{...}, {...}]}
      {"kind":"error","value":"..."}
    Returns: List[{title, url, content, source}]
    """
    docs: List[Dict[str, str]] = []
    for item in evidence or []:
        kind = item.get("kind")
        src = item.get("source") or item.get("tool") or "tool"
        val = item.get("value")

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
                    for m in meetings:
                        title = m.get("title", "Untitled")
                        start = m.get("scheduled_start_time") or m.get("start_time")
                        end = m.get("scheduled_end_time") or m.get("end_time")
                        mtype = (m.get("meeting_type") or "").capitalize()
                        url = m.get("meeting_url") or m.get("share_url", "")

                        # Summary (use default_summary if present, else build fallback)
                        summary_data = m.get("default_summary")
                        if isinstance(summary_data, dict):
                            summary_text = summary_data.get("markdown_formatted") or summary_data.get("text")
                        else:
                            summary_text = summary_data or ""

                        if not summary_text:
                            invitees = [
                                i.get("name") for i in m.get("calendar_invitees", [])
                                if isinstance(i, dict) and i.get("is_external")
                            ]
                            if invitees:
                                summary_text = f"Meeting with external participants: {', '.join(invitees)}"
                            else:
                                summary_text = "No meeting summary available."

                        # Action items
                        actions_data = m.get("action_items") or []
                        action_texts = []
                        for a in actions_data:
                            if isinstance(a, dict):
                                desc = a.get("description")
                                if desc:
                                    action_texts.append(desc)

                        bullet = f"- **{title}** ({mtype})\n  {start} → {end} UTC"
                        if url:
                            bullet += f"\n  [Join]({url})"
                        bullet += f"\n  _Summary_: {summary_text}"
                        if action_texts:
                            bullet += "\n  _Action Items_: " + "; ".join(action_texts)

                        summaries.append(bullet)

                    compact = "\n".join(summaries)
                except Exception:
                    compact = _json.dumps(val, ensure_ascii=False)[:json_len]

            else:
                # Default for all other JSON results
                try:
                    compact = _json.dumps(val, ensure_ascii=False)[:json_len]
                except Exception:
                    compact = str(val)[:json_len]

            docs.append({
                "title": f"{src} (json)",
                "url": "",
                "content": compact,
                "source": src
            })
        elif kind == "error":
            docs.append({
                "title": f"{src} (error)", "url": "", "content": str(val), "source": src
            })
        else:
            docs.append({
                "title": f"{src} ({kind or 'unknown'})", "url": "", "content": str(val)[:text_len], "source": src
            })

    return docs[:max_docs]

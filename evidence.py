# evidence.py
from typing import Any, Dict, List

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
            import json as _json
            try:
                compact = _json.dumps(val, ensure_ascii=False)[:json_len]
            except Exception:
                compact = str(val)[:json_len]
            docs.append({
                "title": f"{src} (json)", "url": "", "content": compact, "source": src
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

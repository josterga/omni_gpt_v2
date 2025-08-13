import re
from typing import Any, Dict, List

def strip_code_fences(txt: str) -> str:
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def collect_arg_refs(args: Dict[str, Any]) -> List[str]:
    out = []
    if isinstance(args, dict):
        if "$ref" in args and isinstance(args["$ref"], str):
            out.append(args["$ref"])
        for v in args.values():
            out.extend(collect_arg_refs(v))
    elif isinstance(args, list):
        for v in args:
            out.extend(collect_arg_refs(v))
    return out

_STEP_REF_RE = re.compile(r"([^\.]+)\.output\[")

def step_deps(step_id: str, args: Dict[str, Any]) -> List[str]:
    deps = []
    for ref in collect_arg_refs(args):
        m = _STEP_REF_RE.match(ref)
        if m:
            deps.append(m.group(1))
    return deps

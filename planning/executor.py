import time
from typing import Any, Dict, List
from .types import ToolCatalog, PlanStep, ToolResult
from .utils import step_deps

def _resolve_ref(ref: str, prior_outputs: Dict[str, Dict[str, Any]]):
    import re
    m = re.match(r"([^\.]+)\.output\[(.*)\]$", ref)
    if not m:
        return None
    sid, path = m.group(1), m.group(2)
    cur = prior_outputs.get(sid, {}).get("output", {})
    # tiny jsonpath-lite
    tokens, buf, i = [], "", 0
    while i < len(path):
        c = path[i]
        if c == '.':
            if buf: tokens.append(buf); buf = ""
            i += 1
        elif c == '[':
            if buf: tokens.append(buf); buf = ""
            j = path.find(']', i+1)
            tokens.append(int(path[i+1:j]))
            i = j + 1
        else:
            buf += c; i += 1
    if buf: tokens.append(buf)
    for t in tokens:
        try:
            cur = cur[t]
        except Exception:
            return None
    return cur

def _materialize_args(args: Any, prior: Dict[str, Dict[str, Any]]):
    if isinstance(args, dict):
        if "$ref" in args and isinstance(args["$ref"], str):
            return _resolve_ref(args["$ref"], prior)
        return {k: _materialize_args(v, prior) for k, v in args.items()}
    if isinstance(args, list):
        return [_materialize_args(v, prior) for v in args]
    return args

def _topological_batches(steps: List[PlanStep]) -> List[List[PlanStep]]:
    id_to_step = {s["id"]: s for s in steps}
    deps = {s["id"]: set(step_deps(s["id"], s.get("args", {}))) for s in steps}
    indeg = {sid: len(d) for sid, d in deps.items()}
    ready = [sid for sid, d in indeg.items() if d == 0]
    batches, visited = [], set()
    while ready:
        batch = [id_to_step[sid] for sid in ready]
        batches.append(batch)
        visited.update(ready)
        # recompute ready
        next_ready = []
        for sid, dset in deps.items():
            if sid in visited: 
                continue
            deps[sid] = {d for d in dset if d not in visited}
            if not deps[sid]:
                next_ready.append(sid)
        ready = next_ready
    if len(visited) != len(steps):  # cycle fallback
        leftover = [s for s in steps if s["id"] not in visited]
        batches.append(leftover)
    return batches

class ToolExecutor:
    def __init__(self, tool_catalog: ToolCatalog, logger=None):
        self.tools = tool_catalog
        self.logger = logger

    def run(self, steps: List[PlanStep]):
        trace: Dict[str, Dict[str, Any]] = {}
        prior: Dict[str, Dict[str, Any]] = {}
        for batch in _topological_batches(steps):
            for step in batch:
                sid, tool, args = step["id"], step["tool"], step.get("args", {})
                spec = self.tools.get(tool)
                start = time.time()
                if not spec:
                    trace[sid] = {"status": "error", "error": f"Unknown tool: {tool}", "elapsed_ms": 0}
                    prior[sid] = {"status": "error", "output": {}}
                    continue
                try:
                    mat_args = _materialize_args(args, prior)
                    result: ToolResult = spec["run"](mat_args)
                    elapsed = int((time.time() - start) * 1000)
                    trace[sid] = {"status": "ok", "tool": tool, "args": mat_args,
                                  "output": result, "elapsed_ms": elapsed}
                    prior[sid] = {"status": "ok", "output": result}
                except Exception as e:
                    elapsed = int((time.time() - start) * 1000)
                    trace[sid] = {"status": "error", "tool": tool, "args": args,
                                  "error": str(e), "elapsed_ms": elapsed}
                    prior[sid] = {"status": "error", "output": {}}

        # flatten to evidence for synthesis
        evidence = []
        for sid, rec in trace.items():
            if rec.get("status") != "ok": 
                continue
            out = rec.get("output", {})
            kind, val = out.get("kind"), out.get("value")
            if kind == "docs":
                evidence.extend(val if isinstance(val, list) else [])
            else:
                preview = out.get("preview") or (str(val)[:280] if val is not None else "")
                evidence.append({"text": preview, "source": rec["tool"]})
        return trace, evidence

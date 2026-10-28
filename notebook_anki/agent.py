import asyncio
import json
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    ServerToolUseBlock,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)
from pydantic import ValidationError
from scipy.optimize import linear_sum_assignment

from . import anki, ocr
from .models import PageProposal, note_key

PROMPT = (Path(__file__).parent / "prompt.md").read_text()
SERVER = "nb"


def _inline_refs(schema: dict) -> dict:
    """Tool input schemas are safest without $ref/$defs."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _gap(a: list[int], b: list[int]) -> float:
    """Distance between two rectangles' edges (0 when they touch or overlap)."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def _text(data) -> dict:
    return {"content": [{"type": "text", "text": data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}]}


def _error(msg: str) -> dict:
    return {**_text(msg), "is_error": True}


class Job:
    def __init__(self, page_dir: Path, page_file: str):
        self.dir = page_dir
        self.page_file = page_file
        self.img = cv2.imread(str(page_dir / page_file))
        self.boxes = {b["id"]: b["bbox"] for b in json.loads((page_dir / "boxes.json").read_text())}
        self.lines: list[dict] = []
        self.proposal: PageProposal | None = None

    def tools(self):
        @tool(
            "ocr",
            "Run PaddleOCR on the page (or a region). Returns lines with id, lang, text, score, bbox.",
            {
                "type": "object",
                "properties": {
                    "langs": {"type": "array", "items": {"type": "string"}, "description": "PaddleOCR language codes"},
                    "mask_boxes": {"type": "array", "items": {"type": "string"}, "description": "Sketch box ids to blank out"},
                    "region": {"type": "array", "items": {"type": "integer"}, "description": "[x1, y1, x2, y2] to read only part of the page"},
                },
                "required": ["langs"],
            },
        )
        async def ocr_tool(args):
            unknown = [b for b in args.get("mask_boxes", []) if b not in self.boxes]
            if unknown:
                return _error(f"Unknown box ids: {unknown}")
            mask = [self.boxes[b] for b in args.get("mask_boxes", [])]
            try:
                lines = await asyncio.to_thread(ocr.run, self.img, args["langs"], args.get("region"), mask)
            except Exception as e:  # unknown language codes etc.
                return _error(f"OCR failed: {e}")
            if not args.get("region"):
                self.lines = lines
            return _text(lines)

        @tool(
            "layout_hints",
            "For each sketch box: nearest OCR lines (from the last full-page ocr call) by edge distance "
            "in pixels, plus a one-to-one assignment suggestion minimizing total distance.",
            {
                "type": "object",
                "properties": {"sketch_boxes": {"type": "array", "items": {"type": "string"}}},
                "required": ["sketch_boxes"],
            },
        )
        async def layout_hints(args):
            if not self.lines:
                return _error("Run ocr on the full page first.")
            ids = [b for b in args["sketch_boxes"] if b in self.boxes]
            cost = np.array([[_gap(self.boxes[b], l["bbox"]) for l in self.lines] for b in ids])
            rows, cols = linear_sum_assignment(cost) if len(ids) else ([], [])
            suggestion = {ids[r]: self.lines[c]["id"] for r, c in zip(rows, cols)}
            hints = {
                b: [
                    {"line": self.lines[j]["id"], "text": self.lines[j]["text"], "gap": round(cost[i, j])}
                    for j in np.argsort(cost[i])[:4]
                ]
                for i, b in enumerate(ids)
            }
            return _text({"nearest": hints, "assignment": suggestion})

        @tool(
            "anki_lookup",
            "Find existing Anki notes for these cards (matched by language + kind + normalized key).",
            {
                "type": "object",
                "properties": {
                    "target_lang": {"type": "string"},
                    "cards": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"kind": {"enum": ["vocab", "sentence"]}, "key": {"type": "string"}},
                            "required": ["kind", "key"],
                        },
                    },
                },
                "required": ["target_lang", "cards"],
            },
        )
        async def anki_lookup(args):
            from .models import Card

            keys = [
                note_key(args["target_lang"], Card(kind=c["kind"], key=c["key"], front="", translation="", production=False))
                for c in args["cards"]
            ]
            try:
                found = await asyncio.to_thread(anki.find_existing, keys)
            except anki.AnkiError as e:
                return _error(str(e))
            return _text({k: {"deck": v["deck"], "fields": v["fields"]} for k, v in found.items()} or "No existing notes.")

        @tool("submit_proposal", "Submit the final card proposal for review.", _inline_refs(PageProposal.model_json_schema()))
        async def submit_proposal(args):
            try:
                proposal = PageProposal.model_validate(args)
            except ValidationError as e:
                return _error(str(e))
            errors = []
            if not self.lines:
                # reading by eye alone misses things OCR gets right (e.g. pinyin tones)
                errors.append("Run ocr on the full page first and compare it with your reading.")
            labels = {b.id: b.kind for b in proposal.boxes}
            if missing := set(self.boxes) - set(labels):
                errors.append(f"Label every candidate box; missing: {sorted(missing)}")
            used = [c.image for c in proposal.cards if c.image]
            for img in used:
                if labels.get(img) != "sketch":
                    errors.append(f"{img} is used as image but not labelled sketch")
            if dupes := {i for i in used if used.count(i) > 1}:
                errors.append(f"Each sketch belongs to at most one card: {sorted(dupes)}")
            keys = {c.key for c in proposal.cards}
            for f in proposal.flags:
                if f.card is not None and f.card not in keys:
                    errors.append(f"Flag refers to unknown card key {f.card!r}: {f.message}")
            if errors:
                return _error("\n".join(errors))
            self.proposal = proposal
            (self.dir / "proposal.json").write_text(proposal.model_dump_json(indent=2))
            return _text(f"Saved: {len(proposal.cards)} cards for deck {proposal.deck()}.")

        return [ocr_tool, layout_hints, anki_lookup, submit_proposal]


def _preview(src: Path, dst: Path, max_side: int = 1568) -> float:
    """JPEG copy at the size Claude actually looks at; returns the scale factor."""
    img = cv2.imread(str(src))
    scale = min(1.0, max_side / max(img.shape[:2]))
    cv2.imwrite(str(dst), cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return scale


async def run(page_dir: Path, page_file: str, model: str = "sonnet", on_event: Callable[[dict], None] = print) -> PageProposal:
    job = Job(page_dir, page_file)
    server = create_sdk_mcp_server(SERVER, tools=job.tools())
    tool_names = [f"mcp__{SERVER}__{t}" for t in ("ocr", "layout_hints", "anki_lookup", "submit_proposal")]
    h, w = job.img.shape[:2]
    scale = _preview(page_dir / page_file, page_dir / "claude_page.jpg")
    _preview(page_dir / "annotated.png", page_dir / "claude_annotated.jpg")
    options = ClaudeAgentOptions(
        system_prompt=PROMPT,
        model=model,
        cwd=str(page_dir),
        tools=["Read", "WebSearch"],
        allowed_tools=["Read", "WebSearch", *tool_names],
        mcp_servers={SERVER: server},
        max_turns=40,
        max_buffer_size=20 * 1024 * 1024,
    )
    prompt = (
        f"page: {page_dir / 'claude_page.jpg'}\n"
        f"annotated: {page_dir / 'claude_annotated.jpg'}\n"
        f"Both images are shown at {scale:.2f}x of the original {w}x{h} px page; all coordinates "
        f"(boxes, OCR, regions) are in original pixels.\n"
        f"candidate boxes: {json.dumps(job.boxes)}"
    )
    async with ClaudeSDKClient(options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        on_event({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        name = block.name.removeprefix(f"mcp__{SERVER}__")
                        detail = block.input.get("file_path") or block.input.get("langs") or block.input.get("query") or ""
                        on_event({"type": "tool", "name": name, "detail": str(detail)})
                    elif isinstance(block, ServerToolUseBlock):
                        on_event({"type": "tool", "name": block.name, "detail": str(block.input.get("query", ""))})
            elif isinstance(msg, ResultMessage):
                on_event({"type": "done", "cost_usd": msg.total_cost_usd, "turns": msg.num_turns, "error": msg.is_error})
    if job.proposal is None:
        raise RuntimeError("Claude finished without submitting a proposal.")
    return job.proposal


if __name__ == "__main__":
    from .extract import extract

    src = Path(sys.argv[1])
    page_dir = Path("data/pages") / src.stem
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / src.name).write_bytes(src.read_bytes())
    extract(page_dir / src.name, page_dir)
    proposal = asyncio.run(run(page_dir, src.name, model=sys.argv[2] if len(sys.argv) > 2 else "sonnet"))
    print(proposal.model_dump_json(indent=2))

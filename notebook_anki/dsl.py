"""The DSL backend: a page is one note, a `.txt` file written straight into the DSL data folder,
its cards as DSL flashcards in Markdown (see docs/architecture.md, "DSL")."""

import json
import re
import secrets
from pathlib import Path

import cv2
import numpy as np

from .models import Card, PageProposal, note_key

# the keywords that open DSL's keyword section (DSL item language v13, spec/keywords.yaml there)
KEYWORDS = ("appointment", "recurring", "media", "tasks", "tags", "source", "issue", "daily", "data", "extra", "publish")
# a card line that would end the note's body or the card block, unless indented
_OPENER = re.compile(rf"^(?:(?:{'|'.join(KEYWORDS)})(?:\s|$)|->|\(|::[ \t]*$|---[ \t]*$)")
_STATE = re.compile(r"^media files/flashcards/\S+\.json[ \t]*$")
# as DSL's own image uploads
MAX_EDGE = 1920
BLANK = "___"


def status(dsl_dir: str) -> dict:
    if not dsl_dir:
        return {"ok": False, "error": "Set the DSL folder under Settings."}
    d = Path(dsl_dir).expanduser()
    if not d.is_dir():
        return {"ok": False, "error": f"{d} is not a folder."}
    return {"ok": True, "label": f"DSL: {d.name}"}


def md(text: str) -> str:
    """Learner text as Markdown that shows as written: no emphasis, code, math, HTML or references."""
    text = re.sub(r"([\\*_`$<])", r"\\\1", text).replace("[[", r"\[[")
    return re.sub(r"^(\s{0,3})~~~", r"\1\\~~~", text, flags=re.M)


def guard(line: str) -> str:
    return " " + line if _OPENER.match(line) else line


def _parts(*parts: str) -> list[str]:
    return [line for p in parts if p.strip() for line in p.strip().split("\n")]


def block(front: list[str], back: list[str]) -> str:
    return "\n".join(["::", *map(guard, front), "---", *map(guard, back), "::"])


def _mark(text: str, blanked: bool, hint: str = "") -> str:
    """A run of chunks as the blank (front) or in bold (back); whitespace around it stays outside."""
    core = text.strip()
    lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]
    inner = f"{BLANK} ({md(hint)})" if blanked and hint else BLANK if blanked else f"**{md(core)}**"
    return f"{lead}{inner}{trail}"


def _focus(card: Card, blanked: bool) -> str:
    # adjacent focus chunks make one blank, the first one shows the hint (as anki.render_cloze)
    c, out, run, hint = card.cloze, [], "", card.cloze.hint
    for i, chunk in enumerate(c.chunks):
        if i in c.focus:
            run += chunk
            continue
        if run:
            out.append(_mark(run, blanked, hint))
            run, hint = "", ""
        out.append(md(chunk))
    if run:
        out.append(_mark(run, blanked, hint))
    return "".join(out)


def _build(card: Card, k: int, blanked: bool) -> str:
    chunks = card.cloze.chunks
    return md("".join(chunks[:-k])) + _mark("".join(chunks[-k:]), blanked)


def card_blocks(card: Card, sketch: str | None) -> list[str]:
    """The card's flashcards, laid out as the Anki templates in note_types/."""
    main, translation, notes = f"**{md(card.front)}**", md(card.translation), md(card.notes)
    image = f"![[{sketch}]]" if sketch else ""
    mode = card.cloze.mode if card.cloze else None
    if mode == "focus":
        return [block(_parts(_focus(card, True)), _parts(_focus(card, False), translation, image, notes))]
    out = [block(_parts(main), _parts(translation, image, notes))]
    if mode == "build":
        out += [block(_parts(translation, _build(card, k, True)), _parts(_build(card, k, False), image, notes))
                for k in range(1, len(card.cloze.chunks) + 1)]
    elif card.production:
        out.append(block(_parts(image, translation), _parts(main, notes)))
    return out


def _tag(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _yaml(s: str) -> str:
    return s if re.fullmatch(r"[\w][\w .'-]*", s) else json.dumps(s, ensure_ascii=False)


def render_note(proposal: PageProposal, page_id: str, blocks: list[str], files: list[str], kept: list[str] = ()) -> str:
    """The note's text: the topic as heading, the cards, its files (`kept`: the lines naming
    the cards' review state, as DSL wrote them), tags and a small data block."""
    tags = " ".join(t for t in ("sketchbook", _tag(proposal.target_lang)) if t)
    lines = [
        re.sub(r"\s+", " ", proposal.topic).strip() or "Sketchbook page", "",
        *[line for b in blocks for line in (b, "")],
        *[f"media {f}" for f in files], *kept,
        f"tags {tags}",
        "data",
        f"sketchbook: {_yaml(page_id)}",
        f"language: {_yaml(proposal.target_lang)}",
    ]
    return "\n".join(lines) + "\n"


def _new_name(dsl_dir: Path, topic: str) -> str:
    """As DSL names a new note: `<slug of its first line>-<hex>.txt`."""
    stem = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40].rstrip("-") or "item"
    while (dsl_dir / (name := f"{stem}-{secrets.token_hex(3)}.txt")).exists():
        pass
    return name


def _write_atomic(path: Path, data: bytes) -> None:
    # DSL ignores dotfiles, so it never sees the half-written file
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _webp(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode WebP")
    return buf.tobytes()


def page_image(path: Path) -> bytes:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read {path.name}")
    h, w = img.shape[:2]
    if (scale := MAX_EDGE / max(h, w)) < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return _webp(img, 85)


def sketch_image(path: Path) -> bytes:
    """The sketch crop with its ink inverted for DSL's dark theme, transparency kept, lossless."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read {path.name}")
    img[..., :3] = 255 - img[..., :3]
    return _webp(img, 101)


def write(proposal: PageProposal, page_dir: Path, page_file: str, dsl_dir: str,
          note: str | None, previous_files: list[str]) -> tuple[str, list[str], list[dict]]:
    """Writes the page's note (a new one, or over `note`) and its images.
    Returns the note's file name, the files it names and one result per included card."""
    if not (s := status(dsl_dir))["ok"]:
        raise ValueError(s["error"])
    d = Path(dsl_dir).expanduser()
    (d / "images").mkdir(exist_ok=True)
    note = note or _new_name(d, proposal.topic)
    path, stem = d / note, note.removesuffix(".txt")
    kept = [l.rstrip() for l in path.read_text(encoding="utf-8").split("\n") if _STATE.match(l)] if path.exists() else []
    action = "updated" if path.exists() else "added"

    files = {f"images/{stem}-page.webp": page_image(page_dir / page_file)}
    blocks, results = [], []
    for card in (c for c in proposal.cards if c.include):
        key = note_key(proposal.target_lang, card)
        if card.cloze and (problems := card.cloze.problems(card.front)):
            results.append({"key": key, "card": key, "action": "error", "error": "; ".join(problems)})
            continue
        sketch = None
        if card.image:
            rel = f"images/{stem}-{card.image}.webp"
            if rel not in files:
                files[rel] = sketch_image(page_dir / f"{card.image}.png")
            sketch = Path(rel).name
        blocks += card_blocks(card, sketch)
        results.append({"key": key, "card": key, "action": action, "note": note, "where": proposal.topic,
                        "fields": {"front": card.front, "translation": card.translation, "notes": card.notes}})

    for rel, data in files.items():
        _write_atomic(d / rel, data)
    _write_atomic(path, render_note(proposal, page_dir.name, blocks, list(files), kept).encode("utf-8"))
    # the images an earlier commit wrote that the note no longer names
    for rel in set(previous_files) - set(files):
        if rel.startswith(f"images/{stem}-"):
            (d / rel).unlink(missing_ok=True)
    return note, list(files), results


def _latest_commit(page: Path) -> Path | None:
    if (page / "commit.json").exists():
        return page / "commit.json"
    versions = sorted((p for p in page.glob("versions/*/commit.json") if p.parent.name.isdigit()),
                      key=lambda p: int(p.parent.name))
    return versions[-1] if versions else None


def find_existing(pages: Path, keys: list[str]) -> dict[str, dict]:
    """Cards already in DSL notes, by key, as each page's latest commit wrote them:
    {key: {"where", "page", "note", "front", "translation", "notes"}}."""
    wanted, found = set(keys), {}
    for page in pages.iterdir():
        if not (commit := _latest_commit(page)):
            continue
        for r in json.loads(commit.read_text()):
            if r.get("note") and r["key"] in wanted:
                found[r["key"]] = {"where": r["where"], "page": page.name, "note": r["note"], **r["fields"]}
    return found

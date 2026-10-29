import hashlib
from pathlib import Path

import httpx

from .models import Card, PageProposal, note_key, slug

URL = "http://localhost:8765"
TEMPLATES = Path(__file__).parent / "note_types"

NOTE_TYPES = {
    "vocab": {"name": "NB Vocab", "main": "Term"},
    "sentence": {"name": "NB Sentence", "main": "Sentence"},
    # focus and build notes of a sentence; Build switches the translation onto the front
    "cloze": {"name": "NB Cloze", "main": "Text", "cloze": True},
}


def _fields(t: dict) -> list[str]:
    # Key first: Anki's own duplicate check looks at the first field
    if t.get("cloze"):
        return ["Key", t["main"], "Translation", "Image", "Notes", "Build", "Source", "RelatedKey"]
    return ["Key", t["main"], "Translation", "Image", "Notes", "Production", "Source", "RelatedKey"]


class AnkiError(Exception):
    pass


def invoke(action: str, **params):
    try:
        r = httpx.post(URL, json={"action": action, "version": 6, "params": params}, timeout=30)
    except httpx.ConnectError:
        raise AnkiError(f"AnkiConnect not reachable at {URL}. Is Anki running with the AnkiConnect add-on?")
    body = r.json()
    if body["error"]:
        raise AnkiError(f"{action}: {body['error']}")
    return body["result"]


def status() -> dict:
    try:
        version = invoke("version")
    except AnkiError as e:
        return {"ok": False, "error": str(e)}
    models = set(invoke("modelNames"))
    return {
        "ok": True,
        "version": version,
        "profile": invoke("getActiveProfile"),
        "note_types_installed": all(t["name"] in models for t in NOTE_TYPES.values()),
    }


def _templates(t: dict) -> dict[str, dict[str, str]]:
    def read(name):
        return (TEMPLATES / name).read_text().replace("{{MAIN}}", "{{%s}}" % t["main"])

    if t.get("cloze"):
        return {"Cloze": {"Front": read("cloze.front.html"), "Back": read("cloze.back.html")}}
    return {
        "Recognition": {"Front": read("recognition.front.html"), "Back": read("recognition.back.html")},
        "Production": {"Front": read("production.front.html"), "Back": read("production.back.html")},
    }


def ensure_note_types() -> None:
    """Create the note types, or bring templates, styling and fields up to date."""
    css = (TEMPLATES / "style.css").read_text()
    existing = set(invoke("modelNames"))
    for t in NOTE_TYPES.values():
        name, fields, templates = t["name"], _fields(t), _templates(t)
        if name not in existing:
            invoke(
                "createModel",
                modelName=name,
                inOrderFields=fields,
                css=css,
                isCloze=t.get("cloze", False),
                cardTemplates=[{"Name": n, **sides} for n, sides in templates.items()],
            )
            continue
        have = invoke("modelFieldNames", modelName=name)
        for f in fields:
            if f not in have:
                invoke("modelFieldAdd", modelName=name, fieldName=f, index=len(have))
                have.append(f)
        invoke("updateModelTemplates", model={"name": name, "templates": templates})
        invoke("updateModelStyling", model={"name": name, "css": css})


def _escape(value: str) -> str:
    for ch in '\\"*_:':
        value = value.replace(ch, "\\" + ch)
    return value


def find_existing(keys: list[str]) -> dict[str, dict]:
    """Existing notes by Key: {key: {"id", "fields": {name: value}, "deck"}}."""
    if not keys:
        return {}
    models = " OR ".join(f'"note:{t["name"]}"' for t in NOTE_TYPES.values())
    found = {}
    # chunk to keep the search string reasonable
    for i in range(0, len(keys), 50):
        terms = " OR ".join(f'"Key:{_escape(k)}"' for k in keys[i : i + 50])
        ids = invoke("findNotes", query=f"({models}) ({terms})")
        if not ids:
            continue
        infos = invoke("notesInfo", notes=ids)
        cards = invoke("cardsInfo", cards=[info["cards"][0] for info in infos])
        for info, card in zip(infos, cards):
            fields = {name: f["value"] for name, f in info["fields"].items()}
            found[fields["Key"]] = {
                "id": info["noteId"],
                "cards": info["cards"],
                "fields": fields,
                "deck": card["deckName"],
            }
    return found


def _store_image(path: Path) -> str:
    data = path.read_bytes()
    filename = f"nb_{hashlib.sha1(data).hexdigest()[:16]}.png"
    invoke("storeMediaFile", filename=filename, path=str(path.resolve()))
    return filename


def _cloze(n: int, text: str, hint: str = "") -> str:
    # whitespace around a blank stays visible, so the blank covers just the words
    core = text.strip()
    lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]
    return f"{lead}{{{{c{n}::{core}{'::' + hint if hint else ''}}}}}{trail}"


def render_cloze(card: Card) -> str:
    """Cloze markup for the card's focus or build note."""
    c = card.cloze
    if c.mode == "focus":
        # adjacent focus chunks make one blank; all blanks belong to the one card c1
        out, run, hint = [], "", c.hint
        for i, chunk in enumerate(c.chunks):
            if i in c.focus:
                run += chunk
                continue
            if run:
                out.append(_cloze(1, run, hint))
                run, hint = "", ""
            out.append(chunk)
        if run:
            out.append(_cloze(1, run, hint))
        return "".join(out)
    # build: card k blanks the last k chunks, as nested clozes (Anki 2.1.56+)
    text = ""
    for k, chunk in enumerate(reversed(c.chunks), 1):
        text = _cloze(k, chunk + text)
    return text


def note_specs(proposal: PageProposal, card: Card, page_dir: Path) -> list[dict]:
    """The notes a card is written as: {"key", "type", "fields", "keep", "create"}.
    `keep` are the card ords to leave active; the note's other cards get suspended (Anki doesn't
    delete cards whose front became empty). Notes with `create` False only matter if they exist:
    they belong to another drill mode of the same sentence."""
    lang, mode = proposal.target_lang, card.cloze.mode if card.cloze else None
    fields = {"Translation": card.translation, "Notes": card.notes, "Source": card.source, "RelatedKey": card.related_key}
    if card.image:
        fields["Image"] = f'<img src="{_store_image(page_dir / f"{card.image}.png")}">'
    main = NOTE_TYPES[card.kind]["main"]
    production = card.production and mode is None
    specs = [{
        "key": note_key(lang, card),
        "type": card.kind,
        "fields": {**fields, main: card.front, "Production": "y" if production else ""},
        "keep": set() if mode == "focus" else {0, 1} if production else {0},
        "create": mode != "focus",
    }]
    if card.kind != "sentence":
        return specs
    for m in ("focus", "build"):
        spec = {"key": note_key(lang, card, m), "type": "cloze", "keep": set(), "create": False}
        if m == mode:
            spec["fields"] = {**fields, "Text": render_cloze(card), "Build": "y" if m == "build" else ""}
            spec["keep"] = set(range(len(card.cloze.chunks) if m == "build" else 1))
            spec["create"] = True
        specs.append(spec)
    return specs


def _set_active(note_id: int, keep: set[int]) -> None:
    cards = invoke("cardsInfo", cards=invoke("notesInfo", notes=[note_id])[0]["cards"])
    if on := [c["cardId"] for c in cards if c["ord"] in keep]:
        invoke("unsuspend", cards=on)
    if off := [c["cardId"] for c in cards if c["ord"] not in keep]:
        invoke("suspend", cards=off)


def upsert(proposal: PageProposal, page_dir: Path, deck_root: str = "") -> list[dict]:
    """Add new notes to the page deck; update existing ones in place (their deck and
    review history stay as they are). Returns one result per included card."""
    ensure_note_types()
    deck = proposal.deck(deck_root)
    invoke("createDeck", deck=deck)
    cards = [c for c in proposal.cards if c.include]
    specs = {id(c): note_specs(proposal, c, page_dir) for c in cards}
    existing = find_existing([s["key"] for c in cards for s in specs[id(c)]])
    tags = ["nb", f"nb::{slug(proposal.target_lang)}::{slug(proposal.topic)}"]

    results = []
    for card in cards:
        # `card`: the sentence or vocab note key, which the card is known by
        card_key = note_key(proposal.target_lang, card)
        if card.cloze and (problems := card.cloze.problems(card.front)):
            results.append({"key": card_key, "card": card_key, "action": "error", "error": "; ".join(problems)})
            continue
        card_tags = tags + (["generated"] if card.source == "generated" else [])
        for spec in specs[id(card)]:
            key, note = spec["key"], existing.get(spec["key"])
            try:
                if note and "fields" in spec:
                    invoke("updateNoteFields", note={"id": note["id"], "fields": {"Key": key, **spec["fields"]}})
                    invoke("addTags", notes=[note["id"]], tags=" ".join(card_tags))
                    _set_active(note["id"], spec["keep"])
                    results.append({"key": key, "card": card_key, "action": "updated", "note_id": note["id"], "deck": note["deck"]})
                elif note:
                    # a drill mode the sentence no longer uses
                    _set_active(note["id"], spec["keep"])
                    results.append({"key": key, "card": card_key, "action": "suspended", "note_id": note["id"], "deck": note["deck"]})
                elif spec["create"]:
                    note_id = invoke(
                        "addNote",
                        note={
                            "deckName": deck,
                            "modelName": NOTE_TYPES[spec["type"]]["name"],
                            "fields": {"Key": key, **spec["fields"]},
                            "tags": card_tags,
                            "options": {"allowDuplicate": False},
                        },
                    )
                    results.append({"key": key, "card": card_key, "action": "added", "note_id": note_id, "deck": deck})
            except AnkiError as e:
                results.append({"key": key, "card": card_key, "action": "error", "error": str(e)})
    return results

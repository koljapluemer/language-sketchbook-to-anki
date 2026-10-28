import hashlib
from pathlib import Path

import httpx

from .models import Card, PageProposal, note_key, slug

URL = "http://localhost:8765"
TEMPLATES = Path(__file__).parent / "note_types"

NOTE_TYPES = {
    "vocab": {"name": "NB Vocab", "main": "Term"},
    "sentence": {"name": "NB Sentence", "main": "Sentence"},
}


def _fields(main: str) -> list[str]:
    # Key first: Anki's own duplicate check looks at the first field
    return ["Key", main, "Translation", "Image", "Notes", "Production", "Source", "RelatedKey"]


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


def _templates(main: str) -> dict[str, dict[str, str]]:
    def read(name):
        return (TEMPLATES / name).read_text().replace("{{MAIN}}", "{{%s}}" % main)

    return {
        "Recognition": {"Front": read("recognition.front.html"), "Back": read("recognition.back.html")},
        "Production": {"Front": read("production.front.html"), "Back": read("production.back.html")},
    }


def ensure_note_types() -> None:
    """Create the note types, or bring templates, styling and fields up to date."""
    css = (TEMPLATES / "style.css").read_text()
    existing = set(invoke("modelNames"))
    for t in NOTE_TYPES.values():
        name, fields, templates = t["name"], _fields(t["main"]), _templates(t["main"])
        if name not in existing:
            invoke(
                "createModel",
                modelName=name,
                inOrderFields=fields,
                css=css,
                isCloze=False,
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


def note_fields(proposal: PageProposal, card: Card, page_dir: Path) -> dict[str, str]:
    main = NOTE_TYPES[card.kind]["main"]
    fields = {
        "Key": note_key(proposal.target_lang, card),
        main: card.front,
        "Translation": card.translation,
        "Notes": card.notes,
        "Production": "y" if card.production else "",
        "Source": card.source,
        "RelatedKey": card.related_key,
    }
    if card.image:
        fields["Image"] = f'<img src="{_store_image(page_dir / f"{card.image}.png")}">'
    return fields


def _set_production(note_id: int, enabled: bool) -> None:
    # Anki doesn't delete a card whose front became empty, so an existing production
    # card is suspended instead (and unsuspended when switched back on)
    cards = invoke("notesInfo", notes=[note_id])[0]["cards"]
    production = [c["cardId"] for c in invoke("cardsInfo", cards=cards) if c["ord"] == 1]
    if production:
        invoke("unsuspend" if enabled else "suspend", cards=production)


def upsert(proposal: PageProposal, page_dir: Path, deck_root: str = "") -> list[dict]:
    """Add new notes to the page deck; update existing ones in place (their deck and
    review history stay as they are). Returns one result per included card."""
    ensure_note_types()
    deck = proposal.deck(deck_root)
    invoke("createDeck", deck=deck)
    cards = [c for c in proposal.cards if c.include]
    existing = find_existing([note_key(proposal.target_lang, c) for c in cards])
    tags = ["nb", f"nb::{slug(proposal.target_lang)}::{slug(proposal.topic)}"]

    results = []
    for card in cards:
        fields = note_fields(proposal, card, page_dir)
        card_tags = tags + (["generated"] if card.source == "generated" else [])
        try:
            if note := existing.get(fields["Key"]):
                invoke("updateNoteFields", note={"id": note["id"], "fields": fields})
                invoke("addTags", notes=[note["id"]], tags=" ".join(card_tags))
                _set_production(note["id"], card.production)
                results.append({"key": fields["Key"], "action": "updated", "note_id": note["id"], "deck": note["deck"]})
            else:
                note_id = invoke(
                    "addNote",
                    note={
                        "deckName": deck,
                        "modelName": NOTE_TYPES[card.kind]["name"],
                        "fields": fields,
                        "tags": card_tags,
                        "options": {"allowDuplicate": False},
                    },
                )
                results.append({"key": fields["Key"], "action": "added", "note_id": note_id, "deck": deck})
        except AnkiError as e:
            results.append({"key": fields["Key"], "action": "error", "error": str(e)})
    return results

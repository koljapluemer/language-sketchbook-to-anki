import json

import cv2
import numpy as np

from notebook_anki import dsl
from notebook_anki.models import Card, Cloze, PageProposal


def card(**kw) -> Card:
    return Card(**{"kind": "vocab", "key": "piatto", "front": "il piatto", "translation": "the plate",
                   "production": False, **kw})


def proposal(*cards: Card) -> PageProposal:
    return PageProposal(target_lang="Italian", native_lang="English", topic="Cooking nouns",
                        topic_source="page", boxes=[], cards=list(cards))


def test_recognition_and_production():
    blocks = dsl.card_blocks(card(production=True, notes="masculine"), "n-B3.webp")
    assert blocks == [
        "::\n**il piatto**\n---\nthe plate\n![[n-B3.webp]]\nmasculine\n::",
        "::\n![[n-B3.webp]]\nthe plate\n---\n**il piatto**\nmasculine\n::",
    ]


def test_focus_cloze():
    c = card(kind="sentence", key="Io mangio la pizza.", front="Io mangio la pizza.", translation="I eat pizza.",
             cloze=Cloze(mode="focus", chunks=["Io ", "mangi", "o", " la pizza."], focus=[1, 2], hint="to eat"))
    assert dsl.card_blocks(c, None) == ["::\nIo ___ (to eat) la pizza.\n---\nIo **mangio** la pizza.\nI eat pizza.\n::"]


def test_build_cloze():
    c = card(kind="sentence", key="Io mangio.", front="Io mangio.", translation="I eat.", production=True,
             cloze=Cloze(mode="build", chunks=["Io ", "mangio."]))
    assert dsl.card_blocks(c, None) == [
        "::\n**Io mangio.**\n---\nI eat.\n::",
        "::\nI eat.\nIo ___\n---\nIo **mangio.**\n::",
        "::\nI eat.\n___\n---\n**Io mangio.**\n::",
    ]


def test_escaping_and_guard():
    assert dsl.md("a_b *c* $5 [[x]] <b>") == r"a\_b \*c\* \$5 \[[x]] \<b>"
    blocks = dsl.card_blocks(card(notes="tags are fun\n---\n::\n-> x @y\nplain"), None)
    assert blocks == ["::\n**il piatto**\n---\nthe plate\n tags are fun\n ---\n ::\n -> x @y\nplain\n::"]


def test_note_layout():
    text = dsl.render_note(proposal(card()), "p1", ["::\na\n---\nb\n::"], ["images/n-page.webp"],
                           ["media files/flashcards/n-abc.json"])
    assert text == ("Cooking nouns\n\n::\na\n---\nb\n::\n\nmedia images/n-page.webp\n"
                    "media files/flashcards/n-abc.json\ntags sketchbook italian\ndata\nsketchbook: p1\nlanguage: Italian\n")


def _page(tmp_path):
    page = tmp_path / "pages" / "p1"
    page.mkdir(parents=True)
    cv2.imwrite(str(page / "page.jpg"), np.full((40, 30, 3), 200, np.uint8))
    sketch = np.zeros((10, 10, 4), np.uint8)
    sketch[..., 3] = 255
    sketch[0, 0, 3] = 0
    cv2.imwrite(str(page / "B1.png"), sketch)
    return page


def test_write_and_overwrite(tmp_path):
    page, notes = _page(tmp_path), tmp_path / "notes"
    notes.mkdir()
    name, files, results = dsl.write(proposal(card(image="B1")), page, "page.jpg", str(notes), None, [])
    stem = name.removesuffix(".txt")
    assert files == [f"images/{stem}-page.webp", f"images/{stem}-B1.webp"]
    assert results[0]["action"] == "added" and results[0]["note"] == name
    sketch = cv2.imread(str(notes / files[1]), cv2.IMREAD_UNCHANGED)
    # ink inverted, transparency kept
    assert sketch[5, 5].tolist() == [255, 255, 255, 255] and sketch[0, 0, 3] == 0

    # DSL names a reviewed card's state in the note; overwriting keeps that, drops the unused sketch
    path = notes / name
    path.write_text(path.read_text().replace("tags ", f"media files/flashcards/{stem}-0123456789ab.json\ntags "))
    name2, files2, results2 = dsl.write(proposal(card()), page, "page.jpg", str(notes), name, files)
    assert name2 == name and files2 == files[:1] and results2[0]["action"] == "updated"
    assert f"media files/flashcards/{stem}-0123456789ab.json" in path.read_text()
    assert not (notes / files[1]).exists()

    (page / "commit.json").write_text(json.dumps(results2))
    found = dsl.find_existing(tmp_path / "pages", ["italian|vocab|piatto", "italian|vocab|x"])
    assert found == {"italian|vocab|piatto": {"where": "Cooking nouns", "page": "p1", "note": name,
                                              "front": "il piatto", "translation": "the plate", "notes": ""}}


def test_cloze_problems_are_errors(tmp_path):
    page, notes = _page(tmp_path), tmp_path / "notes"
    notes.mkdir()
    bad = card(kind="sentence", front="Io mangio.", cloze=Cloze(mode="build", chunks=["Io"]))
    _, _, results = dsl.write(proposal(bad), page, "page.jpg", str(notes), None, [])
    assert results[0]["action"] == "error"

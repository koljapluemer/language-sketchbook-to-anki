# How it works

## Pipeline

Each uploaded page gets its own folder in `<data folder>/pages/<id>/` and goes through a single background queue (`server.py`):

| status | what happens | files |
|---|---|---|
| `queued` | waiting for the worker | `page.*`, `meta.json` |
| `extracting` | box detection (`extract.py`) | `B1.png`…, `boxes.json`, `annotated.png` |
| `thinking` | Claude agent (`agent.py`) | `claude_page.jpg`, `claude_annotated.jpg`, `proposal.json` (edited in review), `proposal.agent.json` (as submitted) |
| `ready` | waiting for review in the UI | |
| `committed` | written to Anki (`anki.py`) | `commit.json` |

Pages interrupted by a restart are queued again on startup. Progress is streamed to the UI via server-sent events.

## Box detection

`extract.find_boxes` is tuned for recall: it bridges gaps in hand-drawn outlines at several strengths and keeps the most rectangular variant of each box. Title boxes, boxed labels and letter loops show up as candidates too; the agent labels every candidate (`sketch`, `title`, `label`, `noise`). Sketch crops get the dot grid removed and a transparent background.

Run it on its own for debugging:

```sh
uv run python -m notebook_anki.extract examples/mandarin_components.png   # writes to out/mandarin_components/
```

## The agent

The system prompt is `notebook_anki/prompt.md`. The agent gets the page and the annotated page, plus these tools (in-process MCP server):

- `ocr`: PaddleOCR on the page or a region, with sketch boxes masked out
- `layout_hints`: nearest text lines per sketch and a one-to-one assignment (Hungarian algorithm)
- `anki_lookup`: existing notes for the proposed cards
- `submit_proposal`: validates the proposal (every box labelled, each sketch used at most once, …) and returns errors for the agent to fix

It may also use web search for grammar facts. Doubts go into `flags`, which the UI shows with an optional suggested replacement.

Run the agent without the UI:

```sh
uv run python -m notebook_anki.agent examples/mandarin_components.png sonnet
```

## Anki

Two note types, `NB Vocab` and `NB Sentence`, each with a recognition card and an optional production card (enabled per note via the `Production` field; turning it off later suspends the card). Templates live in `notebook_anki/note_types/`.

A sentence card can instead be drilled as a cloze (`cloze` in the proposal), written as a third note type, `NB Cloze` (an Anki cloze type):

| drill | NB Sentence | NB Cloze |
|---|---|---|
| translate (no cloze) | recognition, optional production | |
| focus | | one card blanking the word the sentence illustrates (`focus` chunks, with `hint`) |
| build | recognition | card k blanks the last k chunks, up to the whole sentence; nested clozes, Anki 2.1.56+ |

The sentence's `chunks` come from Claude, since where words begin and end (inflection, attached prefixes, no spaces) needs the language; code only checks that they join back to the sentence (`Cloze.problems`). In review, switching the drill or editing the sentence asks Claude again (`POST /pages/<id>/cloze`, one Haiku call with the "Cloze" rules from `prompt.md`); chunks can also be merged and split by hand. Cloze notes are keyed `language|focus|…` / `language|build|…` next to the sentence's own note, so each drill keeps its history. Switching drills suspends the cards of the drills no longer used, and build cards beyond the current number of chunks.

Notes are identified by the `Key` field: `language|kind|normalized key`. A card whose key already exists updates that note's fields and leaves its deck alone; new notes go into `<deck root>::<language>::<topic>`. All notes are tagged `nb` and `nb::<language>::<topic>`, generated content additionally `generated`.

## Settings

Stored in `<data folder>/settings.json`, editable in the UI: Claude model, OCR device, AnkiConnect URL, deck root.

The data folder is `data/` in the repo unless set otherwise under *Settings*; that choice is kept in `~/.config/language-sketchbook/config.json`. Changing it moves your pages and settings to the new folder, or, if the new folder already holds app data, switches to it.

## Updating a page

Added notes to a page after processing it? *Update photo* replaces the photo of the same page. Everything but `meta.json` moves to `versions/<n>/`, and `meta.json` records the new `version` and the `previous` proposal. The page then runs through the pipeline again, with these differences:

- The agent gets the previous proposal, as reviewed, in its prompt (see "Updated pages" in `prompt.md`). `submit_proposal` rejects a changed `topic` or `target_lang`, so new cards land in the page's deck.
- Afterwards `server.merge_previous` restores review decisions on cards with the same note key: deselected cards stay deselected, and an edited field (front, translation, notes, production) keeps the edit unless the agent changed that field. Edits are found by comparing the reviewed `proposal.json` with `proposal.agent.json`; without it (bundled demos) only deselections carry over.
- The review marks each card as new on the page, changed (showing the previous values) or unchanged (hidden by default). Cards of the previous version that are missing now are listed separately and can be kept. Nothing is deleted from Anki.

## Demo pages

Each image in `examples/` becomes a demo page (`demo-<name>`) the first time the server starts with a data folder. The results in `examples/<name>/` (box crops, `annotated.png`, `boxes.json`, `proposal.json`) are copied along, so demos are ready to review without calling Claude; an example without them is processed like an upload. To refresh a demo's results, run the agent on it (see above) and copy those files from `data/pages/<name>/`.

### OCR on CPU

Set the OCR device to `cpu` in the settings. On machines without an NVIDIA GPU, replace `paddlepaddle-gpu` with `paddlepaddle` in `pyproject.toml` and drop the `paddle-cu129` index and source, then `uv lock`.

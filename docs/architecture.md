# How it works

## Pipeline

Each uploaded page gets its own folder in `data/pages/<id>/` and goes through a single background queue (`server.py`):

| status | what happens | files |
|---|---|---|
| `queued` | waiting for the worker | `page.*`, `meta.json` |
| `extracting` | box detection (`extract.py`) | `B1.png`…, `boxes.json`, `annotated.png` |
| `thinking` | Claude agent (`agent.py`) | `claude_page.jpg`, `claude_annotated.jpg`, `proposal.json` |
| `ready` | waiting for review in the UI | |
| `committed` | written to Anki (`anki.py`) | `commit.json` |

Pages interrupted by a restart are queued again on startup. Progress is streamed to the UI via server-sent events.

## Box detection

`extract.find_boxes` is tuned for recall: it bridges gaps in hand-drawn outlines at several strengths and keeps the most rectangular variant of each box. Title boxes, boxed labels and letter loops show up as candidates too; the agent labels every candidate (`sketch`, `title`, `label`, `noise`). Sketch crops get the dot grid removed and a transparent background.

Run it on its own for debugging:

```sh
uv run python -m notebook_anki.extract examples/t1.png   # writes to out/t1/
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
uv run python -m notebook_anki.agent examples/t1.png sonnet
```

## Anki

Two note types, `NB Vocab` and `NB Sentence`, each with a recognition card and an optional production card (enabled per note via the `Production` field; turning it off later suspends the card). Templates live in `notebook_anki/note_types/`.

Notes are identified by the `Key` field: `language|kind|normalized key`. A card whose key already exists updates that note's fields and leaves its deck alone; new notes go into `<deck root>::<language>::<topic>`. All notes are tagged `nb` and `nb::<language>::<topic>`, generated content additionally `generated`.

## Settings

Stored in `data/settings.json`, editable in the UI: Claude model, OCR device, AnkiConnect URL, deck root.

### OCR on CPU

Set the OCR device to `cpu` in the settings. On machines without an NVIDIA GPU, replace `paddlepaddle-gpu` with `paddlepaddle` in `pyproject.toml` and drop the `paddle-cu129` index and source, then `uv lock`.

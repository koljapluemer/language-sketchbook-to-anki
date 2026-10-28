import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field


class BoxLabel(BaseModel):
    id: str = Field(description="Box id from the annotated page, e.g. B3")
    kind: Literal["sketch", "title", "label", "noise"] = Field(
        description="sketch: a drawing illustrating vocab; title: page title box; "
        "label: boxed text; noise: not a real box"
    )


class Card(BaseModel):
    kind: Literal["vocab", "sentence"]
    key: str = Field(
        description="Identity of the note within the language: the lemma without article "
        "for vocab (e.g. 'piatto'), the sentence itself for sentences"
    )
    front: str = Field(description="Target-language term (with article/gender marker if usual) or sentence")
    translation: str = Field(description="Translation in the learner's native language")
    notes: str = Field("", description="Short HTML-free notes: gender, plural, usage, sources for checked facts")
    image: str | None = Field(None, description="Id of the sketch box illustrating this card, if any")
    production: bool = Field(description="Also create a native -> target card")
    production_reason: str = Field("", description="One short sentence why/why not production")
    source: Literal["page", "generated"] = "page"
    related_key: str = Field("", description="For generated sentences: key of the vocab card they illustrate")
    include: bool = True


class Flag(BaseModel):
    message: str = Field(description="Question or warning for the learner")
    card: str | None = Field(None, description="`key` of the card this flag is about")
    field: str | None = Field(None, description="Card field the suggestion applies to")
    suggestion: str | None = Field(None, description="Proposed replacement value for that field")


class PageProposal(BaseModel):
    target_lang: str = Field(description="English name of the language learned, e.g. 'Italian', 'Egyptian Arabic'")
    native_lang: str = Field(description="English name of the learner's support language, e.g. 'English'")
    topic: str = Field(description="Page topic, used as deck name")
    topic_source: Literal["page", "generated"]
    boxes: list[BoxLabel]
    cards: list[Card]
    flags: list[Flag] = []

    def deck(self, root: str = "") -> str:
        # the root may itself be nested ("Languages::Sketchbook"); language and topic may not
        parts = [*root.split("::"), self.target_lang, self.topic]
        return "::".join(_deck_part(p) for p in parts if _deck_part(p))


def _deck_part(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("::", ":").replace('"', "")).strip()


def note_key(target_lang: str, card: Card) -> str:
    key = unicodedata.normalize("NFC", card.key).lower()
    key = re.sub(r"\s+", " ", key).strip(" .,!?;:")
    return f"{target_lang.lower()}|{card.kind}|{key}"


def slug(s: str) -> str:
    return re.sub(r"[^\w]+", "_", s.lower()).strip("_")

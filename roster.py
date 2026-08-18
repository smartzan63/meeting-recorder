"""Optional participant roster used as a prior for speaker-name enrichment.

Speech carries nicknames and diminutives ("Dima", "Sasha", "Лёша") while the
people you actually want in the transcript have canonical names. Without a
roster the model transcribes whatever it heard, so a mishearing becomes a
speaker name and two different people can both end up called "Ilya".

The roster is DATA, never code. Nothing in this repo contains real names: the
file lives outside the project (by default next to your transcripts, which is
already a host path you choose) and is gitignored wherever it lands. The tool
must work with no roster at all — every function here degrades to "no roster",
and enrichment then behaves as it did before, minus the roster prior.

File format (see roster.example.json):

    {
      "people": [
        {"name": "Ada Lovelace", "aliases": ["Ada", "Countess"]},
        {"name": "Alan Turing",  "aliases": ["Alan"]}
      ]
    }

"aliases" is optional. Matching is case-insensitive and ignores surrounding
whitespace.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import config

logger = logging.getLogger(__name__)


@dataclass
class Person:
    name: str
    aliases: list[str] = field(default_factory=list)

    def all_forms(self) -> list[str]:
        """Canonical name plus aliases, in the order a reader would expect."""
        return [self.name, *self.aliases]


@dataclass
class Roster:
    people: list[Person] = field(default_factory=list)
    path: str | None = None

    def __bool__(self) -> bool:
        return bool(self.people)

    def resolve(self, spoken: str) -> str | None:
        """Map a name the model returned onto a canonical roster name.

        Matches the canonical name first, then aliases. Returns None when the
        roster does not contain the name — the caller must keep the model's
        own answer rather than forcing it onto the nearest person.
        """
        needle = (spoken or "").strip().casefold()
        if not needle:
            return None
        for person in self.people:
            if person.name.strip().casefold() == needle:
                return person.name
        for person in self.people:
            if any(alias.strip().casefold() == needle for alias in person.aliases):
                return person.name
        return None

    def ambiguous_forms(self) -> set[str]:
        """Name forms that more than one person answers to, case-folded.

        A meeting with two people who both go by the same nickname produces a
        form that identifies neither of them. Any naming that rests on such a
        form is a coin flip no matter how confident the model sounds, so the
        caller refuses to auto-fill it.
        """
        seen: dict[str, int] = {}
        for person in self.people:
            for form in {f.strip().casefold() for f in person.all_forms() if f.strip()}:
                seen[form] = seen.get(form, 0) + 1
        return {form for form, count in seen.items() if count > 1}

    def is_ambiguous(self, spoken: str) -> bool:
        return (spoken or "").strip().casefold() in self.ambiguous_forms()

    def prompt_block(self) -> str:
        """The roster rendered for the enrichment prompt."""
        ambiguous = self.ambiguous_forms()
        lines = []
        for person in self.people:
            if person.aliases:
                shared = [a for a in person.aliases if a.strip().casefold() in ambiguous]
                note = f"  (shared with another participant: {', '.join(shared)})" if shared else ""
                lines.append(f"- {person.name} — also called: {', '.join(person.aliases)}{note}")
            else:
                lines.append(f"- {person.name}")
        return "\n".join(lines)


def roster_path() -> Path:
    """Where the roster is read from.

    ROSTER_PATH wins when set. Otherwise it sits beside the transcripts, which
    already point at a host directory outside this repo — so the default keeps
    real names out of the project without any extra configuration.
    """
    configured = os.getenv("ROSTER_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(config.TRANSCRIPTS_DIR) / "roster.json"


def load_roster() -> Roster:
    """Read the roster, or return an empty one.

    Never raises. A missing file is the normal case, and a malformed file must
    not take enrichment down with it — both degrade to "no roster" and say so
    in the log.
    """
    path = roster_path()
    try:
        if not path.is_file():
            logger.info("roster: no file at %s, enriching without a participant prior", path)
            return Roster()
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("roster: could not read %s (%s), enriching without a participant prior", path, e)
        return Roster()

    entries = raw.get("people") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        logger.warning("roster: %s has no \"people\" list, enriching without a participant prior", path)
        return Roster()

    people: list[Person] = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry.strip()
            aliases: list[str] = []
        elif isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
            raw_aliases = entry.get("aliases") or []
            aliases = [str(a).strip() for a in raw_aliases if str(a).strip()] if isinstance(raw_aliases, list) else []
        else:
            continue
        if name:
            people.append(Person(name=name, aliases=aliases))

    # Log the count, never the names — this log is read in shared terminals.
    logger.info("roster: loaded %d people from %s", len(people), path)
    return Roster(people=people, path=str(path))

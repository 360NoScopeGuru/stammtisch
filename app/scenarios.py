"""Roleplay scenarios. Each sets the scene and gives the tutor an opening line."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    opener: str
    min_level: str = "A1"


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in [
        Scenario(
            key="freies_gespraech",
            title="Free conversation",
            description=(
                "Ein offenes, freundliches Gespräch ohne festes Thema. "
                "Du bist neugierig und fragst nach dem Alltag des Lernenden."
            ),
            opener="Hallo! Schön, dich zu sehen. Wie war denn dein Tag heute?",
        ),
        Scenario(
            key="baeckerei",
            title="At the bakery",
            description=(
                "Du bist Verkäufer in einer deutschen Bäckerei. "
                "Der Lernende ist Kunde und möchte etwas kaufen."
            ),
            opener="Guten Morgen! Was darf es denn sein?",
        ),
        Scenario(
            key="arzttermin",
            title="Doctor's appointment",
            description=(
                "Du bist Hausarzt. Der Lernende ist Patient und fühlt sich nicht gut. "
                "Frage nach Symptomen."
            ),
            opener="Guten Tag, bitte setzen Sie sich. Was führt Sie denn zu mir?",
            min_level="A2",
        ),
        Scenario(
            key="wohnungsbesichtigung",
            title="Apartment viewing",
            description=(
                "Du vermietest eine Wohnung in Berlin. Der Lernende möchte sie "
                "besichtigen und stellt Fragen zu Miete, Lage und Nebenkosten."
            ),
            opener="Hallo, kommen Sie rein! Haben Sie gut hergefunden?",
            min_level="B1",
        ),
        Scenario(
            key="bewerbungsgespraech",
            title="Job interview",
            description=(
                "Du führst ein Bewerbungsgespräch. Der Lernende bewirbt sich. "
                "Frage nach Erfahrung, Stärken und Motivation."
            ),
            opener=(
                "Schön, dass Sie da sind. Erzählen Sie doch kurz etwas über sich."
            ),
            min_level="B1",
        ),
        Scenario(
            key="smalltalk_buero",
            title="Office small talk",
            description=(
                "Ihr seid Kollegen in der Kaffeeküche. Lockeres Gespräch über "
                "Wochenende, Wetter, Projekte."
            ),
            opener="Na, auch schon wach? Wie war dein Wochenende?",
            min_level="A2",
        ),
        Scenario(
            key="behoerdengang",
            title="At the Bürgeramt",
            description=(
                "Du bist Sachbearbeiter im Bürgeramt. Der Lernende möchte sich "
                "anmelden. Sei korrekt, etwas bürokratisch, aber hilfsbereit."
            ),
            opener="Guten Tag. Sie haben einen Termin? Worum geht es denn?",
            min_level="B1",
        ),
    ]
}

DEFAULT = "freies_gespraech"


def get(key: str) -> Scenario:
    return SCENARIOS.get(key, SCENARIOS[DEFAULT])


def listing() -> str:
    rows = [
        f"  {s.key:<22} {s.title:<24} ({s.min_level}+)" for s in SCENARIOS.values()
    ]
    return "\n".join(rows)

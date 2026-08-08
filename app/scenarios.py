"""Session topics.

Each carries two openers: `intro` in English for mentor mode, and `opener` in
German for practice mode.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    opener: str       # German — spoken when entering practice mode
    intro: str        # English — spoken when entering mentor mode
    min_level: str = "A1"


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in [
        Scenario(
            key="grundlagen",
            title="Absolute basics",
            description=(
                "The very first building blocks: greetings, saying your name, "
                "where you are from, and how you are feeling."
            ),
            intro=(
                "Hi, good to see you. I am your German tutor, and we will take "
                "this one small step at a time. Let us start with something you "
                "will use every single day: how to greet someone and say your "
                "name. Ready?"
            ),
            opener="Hallo! Wie heißt du?",
        ),
        Scenario(
            key="freies_gespraech",
            title="Free conversation",
            description=(
                "No fixed topic. Whatever the learner wants to talk about, "
                "everyday life, plans, opinions."
            ),
            intro=(
                "Hi! No fixed topic today, so tell me what you would like to "
                "work on. If nothing comes to mind, I can pick something."
            ),
            opener="Hallo! Schön, dich zu sehen. Wie war denn dein Tag heute?",
        ),
        Scenario(
            key="zahlen_zeit",
            title="Numbers and time",
            description=(
                "Counting, prices, telling the time, days of the week, and "
                "making appointments."
            ),
            intro=(
                "Today we are doing numbers and time. It is not glamorous, but "
                "it is the thing you will need constantly, for prices, for "
                "appointments, for train times. Let us start with counting."
            ),
            opener="Wie spät ist es denn jetzt?",
        ),
        Scenario(
            key="baeckerei",
            title="At the bakery",
            description=(
                "Buying bread and coffee: asking for things politely, "
                "understanding prices, paying."
            ),
            intro=(
                "Today we are going to the bakery. This is one of the first "
                "real conversations you will have in Germany, so it is worth "
                "getting comfortable with. I will teach you the phrases first, "
                "then we will roleplay it."
            ),
            opener="Guten Morgen! Was darf es denn sein?",
        ),
        Scenario(
            key="smalltalk_buero",
            title="Office small talk",
            description=(
                "Casual conversation with colleagues: weekends, weather, "
                "projects, coffee."
            ),
            intro=(
                "Today: small talk at work. Germans have a reputation for being "
                "direct, but the coffee kitchen still runs on small talk. Let me "
                "show you a few openers."
            ),
            opener="Na, auch schon wach? Wie war dein Wochenende?",
            min_level="A2",
        ),
        Scenario(
            key="arzttermin",
            title="Doctor's appointment",
            description=(
                "Describing symptoms, understanding questions about your health, "
                "making an appointment."
            ),
            intro=(
                "Today we are covering the doctor. Not fun, but you want these "
                "words before you need them. We will do body parts and how to "
                "say what hurts."
            ),
            opener="Guten Tag, bitte setzen Sie sich. Was führt Sie denn zu mir?",
            min_level="A2",
        ),
        Scenario(
            key="wohnungsbesichtigung",
            title="Apartment viewing",
            description=(
                "Viewing a flat: asking about rent, extra costs, the "
                "neighbourhood, and the contract."
            ),
            intro=(
                "Today: viewing an apartment. The vocabulary here is very "
                "specific and it costs you real money if you misunderstand it, "
                "so we will go carefully."
            ),
            opener="Hallo, kommen Sie rein! Haben Sie gut hergefunden?",
            min_level="B1",
        ),
        Scenario(
            key="behoerdengang",
            title="At the Bürgeramt",
            description=(
                "Registering your address, forms, appointments, and the very "
                "formal German of public offices."
            ),
            intro=(
                "Today we tackle the Bürgeramt. German officialdom has its own "
                "register, very formal and very fixed. Once you know the "
                "patterns it is actually predictable."
            ),
            opener="Guten Tag. Sie haben einen Termin? Worum geht es denn?",
            min_level="B1",
        ),
        Scenario(
            key="bewerbungsgespraech",
            title="Job interview",
            description=(
                "Talking about your experience, strengths and motivation in a "
                "professional register."
            ),
            intro=(
                "Today: job interviews. We will work on talking about your "
                "experience without sounding like a textbook."
            ),
            opener="Schön, dass Sie da sind. Erzählen Sie doch kurz etwas über sich.",
            min_level="B1",
        ),
    ]
}

DEFAULT = "grundlagen"


def get(key: str) -> Scenario:
    return SCENARIOS.get(key, SCENARIOS[DEFAULT])


def listing() -> str:
    return "\n".join(
        f"  {s.key:<22} {s.title:<24} ({s.min_level}+)" for s in SCENARIOS.values()
    )

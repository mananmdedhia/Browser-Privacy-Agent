"""Few-shot for NAMES (project codenames, client names, internal tools): words with no digits, so the ID model never sees them.

Same prototype network as model.py, different candidates (Capitalised words) and different fake companies. The model has to learn from the
SENTENCE around the example ("Nightfall is the internal codename ..."), because the word alone is also an ordinary word ("The Nightfall
weather forecast"). Train and test companies share no words and no sentence templates.

Run:  python -m privylens.fewshot.words [--steps 1200]
Out:  checkpoints/fewshot_words.pt (the model the app loads), reports/fewshot_words.md
"""
import argparse
import os
import random
import re
import statistics as st
import torch
from faker import Faker
from privylens.evaluation.metrics import evaluate
from privylens.fewshot import baselines as B
from privylens.fewshot.model import detector, train_proto

fake = Faker("en_IN")
WORDS = """Nightfall Helix Orion Cedar Aurora Falcon Zenith Mirage Harbor Summit Ember Glacier Meridian Atlas Phoenix Onyx Quartz Vertex
Nimbus Cobalt Tundra Lantern Compass Horizon Eclipse Monsoon Saffron Basalt Cascade Delta Echo Fjord Garnet Hollow Ivory Juniper Kestrel
Lagoon Marble Nebula Obsidian Pinnacle Quasar Ridge Sierra Thunder Umbra Voyager Willow Xenon Yarrow Zephyr Amber Beacon Cinder Dune
Equinox Flint Gale Haven Indigo Jasper Krypton Lotus Mosaic Nectar Orchid Prism Raven Solstice Talon""".split()
random.Random(7).shuffle(WORDS)
TRAIN_W, TEST_W = WORDS[:48], WORDS[48:]                        # test companies use words training never saw
CUES_TRAIN = ["Project {x}", "codename {x}", "Operation {x}", "the {x} initiative", "{x} launch"]
CUES_TEST = ["Program {x}", "internal codename {x}", "{x} rollout", "the {x} workstream", "stealth project {x}"]
POS_TRAIN = ["We will not announce {cue} before March.", "Please update the budget for {cue}.", "{name} leads {cue}.",
             "The deck for {cue} is attached.", "Keep {cue} inside the team, it is confidential.", "{name} asked for the timeline of {cue}."]
POS_TEST = ["Everything about {cue} must stay internal until the board meets.", "Do not forward anything about {cue} outside the company.",
            "Status of {cue} was shared with {name}.", "{name} signed off on {cue} yesterday.", "Our plan for {cue} is still secret."]
NEG = ["The {x} weather forecast looks clear.", "We watched the {x} festival on {day}.", "{name} flew to {city} on {day}.",
       "The {x} hotel in {city} is booked.", "{x} is a lovely name for a cat.", "Lunch at the {x} cafe on {day}.", "Team {x} won the quiz.",
       "Meeting with {name} in {city} on {day}."]
DAYS = ["Monday", "Tuesday", "Friday", "Sunday", "March", "June", "October"]
WCAND = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Za-z]{2,}(?![A-Za-z0-9])")


def wcand(text):
    """Capitalised words of 3+ letters -> [(start, end)]. ponytail: a lowercase codename is never found; relax if that matters."""
    return [m.span() for m in WCAND.finditer(text)]


def pseudo(rng):
    """Invented name like 'Zarvex' (client / tool names that are not dictionary words)."""
    return rng.choice(["Z", "Q", "V", "K", "Br", "Dr", "Tr", "Sh", "L", "M", "N"]) + "".join(
        rng.choice("aeoiu") + rng.choice(["rv", "x", "l", "n", "r", "m", "ll", "th"]) for _ in range(rng.choice([1, 2])))


def make_company(rng, words, cues, sents):
    return {"names": rng.sample(words, 6) + [pseudo(rng) for _ in range(6)], "cues": rng.sample(cues, 2), "sents": sents}


def rows(c, n, rng):
    out = []
    for _ in range(n):
        parts = []                                               # (sentence, the secret inside it or None)
        for _ in range(rng.randint(1, 2)):
            x = rng.choice(c["names"])
            parts.append((rng.choice(c["sents"]).format(cue=rng.choice(c["cues"]).format(x=x), name=fake.name()), x))
        for _ in range(rng.randint(1, 3)):                       # decoys reuse secret words in harmless sentences
            x = rng.choice(c["names"] + WORDS)
            parts.append((rng.choice(NEG).format(x=x, name=fake.name(), city=fake.city(), day=rng.choice(DAYS)), None))
        rng.shuffle(parts)
        text, spans = "", []
        for sent, x in parts:
            text += " " if text else ""
            if x:
                k = sent.find(x)
                spans.append({"start": len(text) + k, "end": len(text) + k + len(x), "label": "CUSTOM"})
            text += sent
        out.append({"text": text, "spans": spans})
    return out


def merge_adjacent(spans, text):
    """'Blue' + 'Cedar' flagged one space apart -> one span 'Blue Cedar'."""
    out = []
    for s in sorted(spans, key=lambda s: s["start"]):
        if out and text[out[-1]["end"]:s["start"]] == " ":
            out[-1] = {**out[-1], "end": s["end"], "conf": min(out[-1]["conf"], s["conf"])}
        else:
            out.append(dict(s))
    return out


def word_detector(model, pos, neg=(), label="CUSTOM"):
    d = detector(model, pos, neg, label=label, cand=wcand)
    return lambda text: merge_adjacent(d(text), text)


def tokens(items):
    """Examples whose marked span has several words ('Blue Cedar') -> one example per word, sentence kept."""
    return [(t, a + m.start(), a + m.end()) for t, a, b in items for m in WCAND.finditer(t[a:b])]


def exact(sup, label="CUSTOM"):
    """Baseline: hide the example words wherever they appear. What a find-and-replace rule does."""
    rx = re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(t[a:b]) for t, a, b in sup))
    return lambda text: [{"start": m.start(), "end": m.end(), "label": label, "conf": 1.0} for m in rx.finditer(text)]


def split(rs):
    P, N = [], []
    for r in rs:
        gold = {(s["start"], s["end"]) for s in r["spans"]}
        for a, b in wcand(r["text"]):
            (P if (a, b) in gold else N).append((r["text"], a, b))
    return P, N


def main(steps=1200, seeds=5, out="reports/fewshot_words.md"):
    rng = random.Random(0)
    train = [make_company(rng, TRAIN_W, CUES_TRAIN, POS_TRAIN) for _ in range(40)]
    test = [make_company(rng, TEST_W, CUES_TEST, POS_TEST) for _ in range(15)]
    model = train_proto(train, "both", steps, kit=(rows, wcand))
    torch.save(model.state_dict(), "checkpoints/fewshot_words.pt")
    tag = B.tagger() if os.path.exists("checkpoints/tagger.pt") else None
    res = {}
    for ci, c in enumerate(test):
        ev = rows(c, 60, random.Random(1000 + ci))
        pos, neg = split(rows(c, 60, random.Random(2000 + ci)))
        for K in (1, 3, 5):
            for s in range(seeds):
                r = random.Random(100 * s + ci)
                sup, nsup = r.sample(pos, K), r.sample(neg, 5)
                dets = {"exact match of the examples": exact(sup), "few-shot (examples with sentences)": word_detector(model, sup),
                        "few-shot + 5 'not this' examples": word_detector(model, sup, nsup)}
                if tag and s == 0: dets["regular NER (no examples)"] = tag
                for name, d in dets.items():
                    e = evaluate(d, ev)
                    res.setdefault((name, K), []).append((e["micro"]["F1"], e["leaked_span"], e["over_redact"]))
        print(f"company {ci + 1}/{len(test)} done", flush=True)
    ms = lambda v: f"{st.mean(v):.3f} ± {st.pstdev(v):.3f}"
    lines = ["# Few-shot detection of unseen NAMES: project codenames (auto-generated by `python -m privylens.fewshot.words`)\n",
             f"{len(train)} train companies, {len(test)} unseen test companies (new words AND new sentence templates), {seeds} draws each.",
             "Decoys use the same words in harmless sentences (\"The Nightfall weather forecast\"). Lower is better for missed / false.\n"]
    for K in (1, 3, 5):
        lines += [f"\n## K = {K} example(s)\n", "| method | F1 | missed | false redaction |", "|---|---|---|---|"]
        for name in dict.fromkeys(n for n, _ in res):
            if (name, K) in res:
                v = res[(name, K)]
                lines.append(f"| {name} | {ms([x[0] for x in v])} | {ms([x[1] for x in v])} | {ms([x[2] for x in v])} |")
    os.makedirs("reports", exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(); main(a.steps, a.seeds)

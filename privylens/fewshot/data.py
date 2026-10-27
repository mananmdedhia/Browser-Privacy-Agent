"""Fake companies for the few-shot experiment (research question: can a model learn a NEW kind of sensitive ID from a few examples?).

Each company has its own employee-ID format (ORX-72841, EMP_N7Q291, ...) and its own way of naming it ("Staff code", "Crew number").
Train companies and test companies share nothing: formats are random, and the wording ("cues") comes from separate lists,
so a test company is truly unseen. Decoys (order numbers, phones, PAN, dates...) look similar but are NOT sensitive here.

Run:  python -m privylens.fewshot.data   (prints a few companies and sample documents)
"""
import random
import re
import string
from faker import Faker

fake = Faker("en_IN")
CUES_TRAIN = ["Employee ID", "Emp No", "Staff code", "Badge number", "Personnel no", "Worker ID", "HR ref", "Associate ID"]
CUES_TEST = ["Payroll ID", "Team member ID", "Crew number", "Colleague ID", "Member no", "Workforce ID"]
DECOY_CUES = ["Order", "Ticket", "Invoice", "Ref", "SKU", "Tracking no", "Booking", "Case"]
POS = ["{cue}: {id}", "Please update {cue} {id} for {name}.", "{name}'s {cue} is {id}.",
       "Record for {name} ({cue} {id}) was approved.", "Re: access request, {cue} {id}", "{cue} {id} was added to payroll by {name}."]
NEG = ["{cue} {x} is pending.", "Your {cue} {x} shipped on {date}.", "See {cue} {x} for details.", "{name} asked about {cue} {x}."]
OTHER = ["Call {phone} today.", "PAN {pan} is on file.", "Aadhaar {aadhaar} was verified.", "Server {ip} restarted on {date}.", "Dated {date}."]

CAND = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9](?:[A-Za-z0-9_\-/.#]*[A-Za-z0-9])?(?![A-Za-z0-9])")


def candidates(text):
    """ID-like tokens (>=4 chars, at least one digit) -> [(start, end)]. Only these are ever scored by the few-shot model.
    ponytail: an ID with no digit at all is never found; relax the digit rule if that format matters."""
    return [m.span() for m in CAND.finditer(text) if len(m.group()) >= 4 and any(c.isdigit() for c in m.group())]


def make_format(rng):
    """(prefix, separator, body): body is a string of D (digit) / L (letter) slots, at least 3 digits."""
    prefix = "".join(rng.choices(string.ascii_uppercase, k=rng.choice([0, 2, 3, 3, 4])))
    body = "DDDDD"
    while body.count("D") < 3 or len(set(body)) == 1 and rng.random() < .5:
        body = "".join(rng.choices("DDDL", k=rng.randint(5, 8)))
    return prefix, (rng.choice(["-", "_", "", "/", "."]) if prefix else ""), body


def render(fmt, rng):
    prefix, sep, body = fmt
    return prefix + sep + "".join(rng.choice(string.digits if c == "D" else string.ascii_uppercase) for c in body)


def make_company(rng, cues_pool):
    return {"fmt": make_format(rng), "cues": rng.sample(cues_pool, 2), "decoy_fmts": [make_format(rng) for _ in range(2)]}


def _digits(rng, n): return "".join(rng.choices(string.digits, k=n))


def _decoy(c, rng):
    """One harmless sentence containing something ID-like."""
    if rng.random() < .6:
        t, x = rng.choice(NEG), render(rng.choice(c["decoy_fmts"]), rng)
        return t.format(cue=rng.choice(DECOY_CUES), x=x, name=fake.name(), date=fake.date_object().strftime("%d/%m/%Y")), x
    t = rng.choice(OTHER)
    return t.format(phone="+91 " + rng.choice("6789") + _digits(rng, 9), pan="".join(rng.choices(string.ascii_uppercase, k=5)) + _digits(rng, 4) + "K",
                    aadhaar=rng.choice("23456789") + _digits(rng, 11), ip=".".join(str(rng.randint(1, 254)) for _ in range(4)),
                    date=fake.date_object().strftime("%d-%m-%Y")), None


def rows(c, n, rng):
    """n short documents (2-4 sentences) for company c. Gold span label is always CUSTOM (the user-defined category)."""
    out = []
    for _ in range(n):
        parts = []                                               # (sentence, id inside it, is it this company's employee ID?)
        for _ in range(rng.randint(1, 2)):
            i = render(c["fmt"], rng)
            parts.append((rng.choice(POS).format(cue=rng.choice(c["cues"]), id=i, name=fake.name()), i, True))
        parts += [(*_decoy(c, rng), False) for _ in range(rng.randint(1, 3))]
        rng.shuffle(parts)
        text, spans = "", []
        for sent, i, pos in parts:
            text += " " if text else ""
            if pos:
                k = sent.find(i)
                spans.append({"start": len(text) + k, "end": len(text) + k + len(i), "label": "CUSTOM"})
            text += sent
        out.append({"text": text, "spans": spans})
    return out


if __name__ == "__main__":
    rng = random.Random(0)
    for c in [make_company(rng, CUES_TRAIN), make_company(rng, CUES_TEST)]:
        print(c["fmt"], c["cues"], render(c["fmt"], rng))
        for r in rows(c, 2, rng):
            print("  ", r["text"], [r["text"][s["start"]:s["end"]] for s in r["spans"]])
    all_rows = [r for _ in range(30) for r in rows(make_company(rng, CUES_TEST), 20, rng)]
    assert all((s["start"], s["end"]) in candidates(r["text"]) for r in all_rows for s in r["spans"]), "candidate finder misses an ID"
    print("candidate recall 100%")

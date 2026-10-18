STRUCTURED = ["EMAIL", "PHONE", "PAN", "AADHAAR", "URL", "IP_ADDRESS", "DATE", "ID_NUMBER"]
CONTEXTUAL = ["PERSON", "ORGANIZATION", "LOCATION"]  # LOCATION also covers addresses
ENTITIES = STRUCTURED + CONTEXTUAL

TAGS = ["O"] + [f"{p}-{e}" for e in ENTITIES for p in "BI"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = dict(enumerate(TAGS))

if __name__ == "__main__":
    assert len(TAGS) == 1 + 2 * len(ENTITIES) and ID2TAG[TAG2ID["I-PERSON"]] == "I-PERSON"
    print(len(TAGS), "tags OK")

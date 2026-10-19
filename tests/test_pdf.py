import pymupdf
from privylens.docs.pdf import redact_pdf
from privylens.engine.pipeline import Engine


def make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Send the report to Rahul at rahul@example.com. Call +91 98765 43210.")
    page.insert_text((72, 130), "Thanks, Rahul. Contact Infosys for the report.")
    doc.set_metadata({"author": "Rahul Sharma"})
    doc.save(path)


def test_values_are_really_removed(tmp_path):
    src, dst = str(tmp_path / "in.pdf"), str(tmp_path / "out.pdf")
    make_pdf(src)
    report, vault = redact_pdf(src, dst, Engine(), hide=["Rahul"], keep=["Infosys"])   # regex-only engine
    out = pymupdf.open(dst)
    text = out[0].get_text()
    for secret in ("rahul@example.com", "98765 43210", "Rahul"):
        assert secret not in text, secret                  # gone from the file's text layer, everywhere (sweep + hide)
    assert "Infosys" in text and "report" in text          # the rest is untouched, keep list respected
    assert out.metadata["author"] == ""                    # metadata scrubbed
    assert report["still_visible_words"] == []
    assert vault["[EMAIL_1]"] == "rahul@example.com" and "[CUSTOM_1]" in vault

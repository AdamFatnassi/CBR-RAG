"""Click through web/index.html in headless Chromium (needs: pip install playwright; playwright install chromium)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else WEB / "shots"
OUT.mkdir(parents=True, exist_ok=True)
errors = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        errors.append(name)


with sync_playwright() as p:
    b = p.chromium.launch()
    for scheme in ("light", "dark"):
        page = b.new_page(viewport={"width": 1280, "height": 900}, color_scheme=scheme)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
        page.goto((WEB / "index.html").as_uri()); page.wait_for_timeout(400)
        if scheme == "dark":
            page.click("nav.tabs button[data-tab=predict]"); page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / "dark_predict.png"), full_page=True)
            continue
        page.screenshot(path=str(OUT / "overview.png"), full_page=True)
        page.click("#ov-go"); page.wait_for_timeout(400)
        check("typical client predicted", page.inner_text("#p-verdict") != "")
        check("10 neighbours shown", page.locator("#nb-table tr").count() == 11)
        check("what-if table filled", page.locator("#whatif tr").count() > 3)
        page.screenshot(path=str(OUT / "predict_typical.png"), full_page=True)
        # history consistency
        page.fill("#previous", "2"); page.dispatch_event("#previous", "input"); page.wait_for_timeout(200)
        check("poutcome switches away from nonexistent", page.input_value("#poutcome") != "nonexistent")
        page.select_option("#poutcome", "success"); page.dispatch_event("#poutcome", "input")
        page.check("#previously_contacted"); page.dispatch_event("#previously_contacted", "input")
        page.fill("#pdays", "3"); page.dispatch_event("#pdays", "input"); page.wait_for_timeout(300)
        s1 = float(page.inner_text("#p-score"))
        print("   score after previous success:", s1, page.inner_text("#p-verdict"))
        # period change
        page.select_option("#period", label=page.locator("#period option").nth(15).inner_text()); page.wait_for_timeout(300)
        print("   period:", page.locator("#period option").nth(15).inner_text(), "score", page.inner_text("#p-score"), "month", page.input_value("#month"))
        # real client + revise/retain
        page.select_option("#fill-pick", "5"); page.wait_for_timeout(300)
        n0 = page.inner_text("#cb-line")
        page.click("#o-reveal"); page.click("#o-confirm"); page.wait_for_timeout(200)
        check("revise result shown", "Revise" in page.inner_text("#o-result"))
        check("case base grew", "1 added" in page.inner_text("#cb-line"))
        page.click("#again"); page.wait_for_timeout(300)
        first = page.locator("#nb-table tr").nth(1).inner_text()
        check("retained client first with similarity 1", "1.000" in first and "added" in first)
        page.click("#nb-all"); check("show all", page.locator("#nb-table tr").count() > 50)
        page.screenshot(path=str(OUT / "predict_after_retain.png"), full_page=True)
        # settings
        page.click("details.card summary")
        page.fill("#s-k", "5"); page.dispatch_event("#s-k", "change"); page.wait_for_timeout(300)
        check("k=5 gives 5 neighbours", page.locator("#nb-table tr").count() == 6)
        page.click("#s-restore"); page.click("#reset"); page.wait_for_timeout(300)
        check("reset", "0 added" in page.inner_text("#cb-line"))
        page.click("#fill-random"); page.wait_for_timeout(300)
        check("random client", page.inner_text("#p-verdict") != "")
        for t in ("results", "how"):
            page.click(f"nav.tabs button[data-tab={t}]"); page.wait_for_timeout(200)
            page.screenshot(path=str(OUT / f"{t}.png"), full_page=True)
        check("results tables", page.locator("#r1-table tr").count() == 6 and page.locator("#r2-table tr").count() == 5)
    ph = b.new_page(viewport={"width": 390, "height": 844})
    ph.on("pageerror", lambda e: errors.append(f"pageerror(phone): {e}"))
    ph.goto((WEB / "index.html").as_uri() + "#predict"); ph.wait_for_timeout(400)
    for t in ("predict", "results", "how", "overview"):
        ph.click(f"nav.tabs button[data-tab={t}]"); ph.wait_for_timeout(150)
        check(f"phone {t}: no sideways scroll", ph.evaluate("document.documentElement.scrollWidth") <= 390)
    ph.click("nav.tabs button[data-tab=predict]"); ph.screenshot(path=str(OUT / "phone_predict.png"))
    b.close()
print("ERRORS:", errors or "none")

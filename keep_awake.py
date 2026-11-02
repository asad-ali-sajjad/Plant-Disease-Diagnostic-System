import os
import sys
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

URL = os.environ.get(
    "APP_URL", "https://plant-disease-diagnostic-system.streamlit.app"
)
SUMMARY = os.environ.get("GITHUB_STEP_SUMMARY")


def log(msg):
    print(msg, flush=True)
    if SUMMARY:  # shows on the run's summary page
        with open(SUMMARY, "a") as f:
            f.write(msg + "\n\n")


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    try:
        resp = page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        log(f"Loaded {URL} (HTTP {resp.status if resp else 'n/a'})")

        btn = page.get_by_role("button", name="Yes, get this app back up!")
        try:
            btn.wait_for(timeout=15000)
            btn.click()
            log("😴 App was asleep, wake button clicked")
            page.wait_for_timeout(30000)
        except PWTimeout:
            log("✅ No wake button, app already awake")

        # stay connected briefly so a real session registers
        page.wait_for_timeout(15000)
        log(f"Page title: {page.title()}")
    except Exception as e:
        log(f"❌ Failed: {e}")
        sys.exit(1)  # marks the run red so you notice
    finally:
        browser.close()

log("Done")
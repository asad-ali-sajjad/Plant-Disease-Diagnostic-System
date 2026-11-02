import os
from playwright.sync_api import sync_playwright

URL = os.environ.get(
    "APP_URL", "https://plant-disease-diagnostic-system.streamlit.app"
)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)

    btn = page.get_by_role("button", name="Yes, get this app back up!")
    try:
        btn.wait_for(timeout=15000)
        btn.click()
        print("App was asleep, woke it")
        page.wait_for_timeout(30000)
    except Exception:
        print("No wake button, app already awake")

    # stay connected briefly so a real session registers
    page.wait_for_timeout(15000)
    browser.close()

print("Done")
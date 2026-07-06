"""
capture_paper_figures.py — one-off script that drives the live morph_app.py
server with a headless browser (Playwright) and saves real screenshots of
the application's current features into paper_figures/, for use in
manuscript_paper.tex.

Not part of the regular pipeline — run manually whenever the UI changes
enough that the paper figures go stale:
    python scripts/capture_paper_figures.py
Requires the Flask server already running on localhost:5001 and
`pip install playwright && playwright install chromium`.
"""
import time

from playwright.sync_api import sync_playwright

BASE = "http://localhost:5001"
OUT = "paper_figures"

HALLSTATT = (47.5622, 13.6493)
VIENNA = (48.2082, 16.3738)


def goto_fresh(page):
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector("#map", timeout=15000)
    page.wait_for_timeout(1500)


def click_lens(page, lat, lon):
    page.evaluate(
        "async ([lat, lon]) => { setMode('lens'); map.setView([lat, lon], 16); "
        "await new Promise(r => setTimeout(r, 400)); await onLensClick(lat, lon); }",
        [lat, lon],
    )
    page.wait_for_timeout(1500)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1050}, device_scale_factor=2)

        # 1. Lens mode: NL description + character fingerprint + influential POIs (Hallstatt)
        goto_fresh(page)
        click_lens(page, *HALLSTATT)
        page.wait_for_timeout(1000)
        page.screenshot(path=f"{OUT}/fig_lens_hallstatt.png")
        print("saved fig_lens_hallstatt.png")

        # Sidebar-only crop for the NL+POI panel
        sidebar = page.query_selector("#sidebar")
        if sidebar:
            sidebar.screenshot(path=f"{OUT}/fig_lens_sidebar.png")
            print("saved fig_lens_sidebar.png")

        # 2. Attention graph overlay (same location)
        page.click("#attn-graph-btn")
        page.wait_for_timeout(1200)
        page.screenshot(path=f"{OUT}/fig_attention_graph.png")
        print("saved fig_attention_graph.png")
        page.click("#attn-graph-btn")  # turn back off
        page.wait_for_timeout(300)

        # 3. Character / Semantic Layer — Austria-wide overview
        goto_fresh(page)
        page.evaluate("map.setView([47.7, 13.4], 8);")
        page.wait_for_timeout(600)
        page.click("#charlayer-btn")
        page.wait_for_timeout(2500)
        page.screenshot(path=f"{OUT}/fig_charlayer_overview.png")
        print("saved fig_charlayer_overview.png")

        # 4. Character / Semantic Layer — zoomed into Vienna (viewport-adaptive re-render)
        page.evaluate("map.setView([48.2082, 16.3738], 13);")
        page.wait_for_timeout(1800)
        page.screenshot(path=f"{OUT}/fig_charlayer_vienna.png")
        print("saved fig_charlayer_vienna.png")
        page.click("#charlayer-btn")
        page.wait_for_timeout(300)

        # 5. Sandbox counterfactual brushing tool
        goto_fresh(page)
        page.evaluate(
            "async ([lat, lon]) => { setMode('sandbox'); map.setView([lat, lon], 15); "
            "await new Promise(r => setTimeout(r, 400)); await onSandboxClick(lat, lon); }",
            [HALLSTATT[0], HALLSTATT[1]],
        )
        page.wait_for_timeout(1200)
        page.click("#sb-brush-add-btn")
        page.wait_for_timeout(500)
        page.screenshot(path=f"{OUT}/fig_sandbox_brush.png")
        print("saved fig_sandbox_brush.png")

        # 6. Semantic Embedding Landscape (UMAP) with multi-select legend filter
        goto_fresh(page)
        click_lens(page, *VIENNA)
        page.click("#umap-expand-btn")
        page.wait_for_timeout(800)
        page.evaluate("toggleUmapDimFilter('Urban')")
        page.wait_for_timeout(200)
        page.evaluate("toggleUmapDimFilter('Nature')")
        page.wait_for_timeout(600)
        page.screenshot(path=f"{OUT}/fig_umap_legend_filter.png")
        print("saved fig_umap_legend_filter.png")
        page.click("#umap-modal .umap-leg-all-btn, #umap-leg-all-btn-modal")
        page.evaluate("closeUmapModal()")

        # 7. History / Time Machine
        goto_fresh(page)
        page.click("#tab-history")
        page.wait_for_timeout(500)
        page.fill("#history-search", "Hallstatt")
        page.wait_for_timeout(900)
        page.keyboard.press("Enter")
        page.wait_for_timeout(2500)
        page.screenshot(path=f"{OUT}/fig_history_timemachine.png")
        print("saved fig_history_timemachine.png")

        browser.close()


if __name__ == "__main__":
    main()

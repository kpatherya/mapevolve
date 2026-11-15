"""Browser checks for the offline results report; no model/API calls."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright, expect
from jev_router.dataset import ROOT


def main():
    output=ROOT/'reports';page_errors=[];external=[];checks=[]
    def passed(name): checks.append(name)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda e:page_errors.append(str(e)))
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
        url=(output/'mapevolve-results.html').as_uri()
        page.goto(url)
        expect(page.locator('#stats .stat').nth(0)).to_have_text('100')
        expect(page.locator('#stats .stat').nth(1)).to_have_text('48%')
        expect(page.locator('#stats .stat').nth(2)).to_have_text('55%')
        passed('Overall figures match the completed benchmark')
        page.locator('#type').select_option('factual_retrieval')
        expect(page.locator('#stats .stat').nth(0)).to_have_text('50')
        expect(page.locator('#stats .stat').nth(1)).to_have_text('84%')
        page.locator('#mostly').click()
        expect(page.locator('#stats .stat').nth(1)).to_have_text('100%')
        passed('Prompt filter and score threshold update charts and metrics')
        page.locator('#reset').click();page.locator('#strict').click()
        page.locator('#overlap').select_option('no')
        expect(page.locator('#stats .stat').nth(0)).to_have_text('66')
        page.locator('#reset').click()
        page.locator('#search').fill('this will never match XYZ123')
        expect(page.locator('#stats .stat').nth(0)).to_have_text('0')
        page.locator('#reset').click()
        passed('Overlap, search, empty state and reset work')
        page.locator('#theme').click()
        page.screenshot(path=str(output/'dashboard-overview.png'),full_page=True)
        page.locator('[data-browse="jev_adaptive"]').first.click()
        expect(page.locator('#cases')).to_be_visible()
        expect(page.locator('#focus')).to_have_value('jev_adaptive')
        expect(page.locator('.answer-card')).to_have_count(3)
        expect(page.locator('#case-detail img')).to_be_visible()
        assert page.locator('#case-detail img').evaluate('(img)=>img.complete && img.naturalWidth>0')
        passed('Chart selection opens the case browser with all three answers and a loaded image')
        first=page.locator('#case-detail .eyebrow').inner_text()
        page.locator('#next').click()
        assert page.locator('#case-detail .eyebrow').inner_text()!=first
        page.locator('#prev').click()
        expect(page.locator('#case-detail .eyebrow')).to_have_text(first)
        page.locator('#enlarge').click();expect(page.locator('#image-modal')).to_be_visible();page.locator('#close-modal').click()
        passed('Previous/next navigation and image enlargement work')
        page.locator('#outcome').select_option('adaptive-win')
        count=page.locator('.case-item').count();assert count>0
        expect(page.locator('.answer-card.focused')).to_have_count(1)
        with page.expect_download() as download:
            page.locator('#export').click()
        assert download.value.suggested_filename.endswith('.csv')
        page.locator('#reset').click()
        page.locator('#search').fill('MEI-FR-031')
        expect(page.locator('.case-item')).to_have_count(1)
        page.locator('#case-detail .diagnostics > summary').click()
        expect(page.locator('.diagnostics-body')).to_be_visible()
        page.screenshot(path=str(output/'dashboard-case.png'),full_page=True)
        with page.expect_download() as download:
            page.locator('#export-case').click()
        assert download.value.suggested_filename=='MEI-FR-031.json'
        passed('Outcome filtering, CSV/JSON exports and routing diagnostics work')
        page.locator('#reset').click();page.locator('#tab-overview').click()
        page.locator('[data-score="4"][data-strategy="strongest"]').click()
        expect(page.locator('#score-filter')).to_be_visible()
        assert page.locator('.case-item').count()==55
        page.locator('#clear-score').click()
        expect(page.locator('.case-item')).to_have_count(100)
        page.locator('#tab-overview').click()
        page.locator('[data-heat="jev_adaptive"][data-difficulty="hard"]').click()
        expect(page.locator('#difficulty')).to_have_value('hard')
        passed('Score segments and difficulty heatmap drill into the correct cases')
        page.locator('#tab-details').click()
        expect(page.locator('#usage-table tr')).to_have_count(4)
        page.goto(url+'#view=cases&case=MEI-FR-050')
        expect(page.locator('#case-detail .eyebrow')).to_have_text('MEI-FR-050')
        passed('Run details and deep links work')
        page.set_viewport_size({'width':390,'height':844})
        page.locator('#tab-overview').click();page.locator('#reset').click()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(output/'dashboard-mobile.png'),full_page=True)
        page.locator('#tab-cases').click()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        passed('Mobile overview and case browser have no horizontal overflow')
        browser.close()
    assert not page_errors,page_errors
    assert not external,external
    result={'status':'passed','checks':checks,'javascript_errors':page_errors,'external_network_requests':external}
    (output/'dashboard-qa.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

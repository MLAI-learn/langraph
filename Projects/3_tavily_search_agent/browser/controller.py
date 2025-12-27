from playwright.sync_api import sync_playwright

class BrowserController:
    def __init__(self, headless=True):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()

    def open_url(self, url: str):
        self.page.goto(url, timeout=60000)
        return f"Opened {url}"

    def click(self, selector: str):
        self.page.click(selector, timeout=10000)
        return f"Clicked {selector}"

    def type_text(self, selector: str, text: str):
        self.page.wait_for_selector(selector, timeout=10000)
        self.page.fill(selector, text)
        return f"Typed text into {selector}"

    def scroll(self, amount: int):
        self.page.mouse.wheel(0, amount)
        return f"Scrolled by {amount}px"

    def extract_text(self):
        return self.page.inner_text("body")

    def close(self):
        self.browser.close()
        self.playwright.stop()
    
    def press_key(self, key: str):
        self.page.keyboard.press(key)
        return f"Pressed key {key}"
    
    def wait_for_selector(self, selector: str):
        self.page.wait_for_selector(selector, timeout=10000)
        return f"Selector {selector} is visible"
    
    def try_accept_consent(self):
        buttons = [
         "button:has-text('I agree')",
          "button:has-text('Accept all')",
          "button:has-text('Accept')"
      ]
        for btn in buttons:
            try:
                self.page.click(btn, timeout=3000)
                return "Accepted consent"
            except:
                pass
        return "No consent dialog"
    
    def type_with_keyboard(self, text: str):
        self.page.keyboard.type(text, delay=50)
        return f"Typed text using keyboard: {text}"
    
    def wait_for_navigation(self):
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            return "Page load event completed"
        except:
            return "Page load wait skipped (continuing)"

    def extract_search_results(self):
        # Bing search results container
        selectors = ["#b_results", "main", "body"]

        for selector in selectors:
            try:
                # Wait briefly for results to appear
                self.page.wait_for_selector(selector, timeout=8000)
                text = self.page.inner_text(selector).strip()
                if text:
                    return text
            except Exception:
                pass

        return ""






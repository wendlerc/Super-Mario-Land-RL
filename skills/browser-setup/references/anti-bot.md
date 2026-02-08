# Anti-Bot Detection Evasion

## The Problem

Sites like X (Twitter), Instagram, LinkedIn actively detect and block headless browsers:

**Detection methods:**
- `navigator.webdriver === true` (Chrome flag)
- Missing plugins/fonts
- Perfect timing (no human reaction delays)
- Automation flags in user agent
- No mouse movement trails
- CAPTCHA challenges (hCaptcha, reCAPTCHA v3)

## Workarounds

### 1. puppeteer-extra-plugin-stealth

Hides automation flags and patches headless detection:

```bash
npm install puppeteer-extra puppeteer-extra-plugin-stealth
```

```javascript
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');

puppeteer.use(StealthPlugin());

const browser = await puppeteer.launch({
  executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox']
});
```

### 2. Human-like Behavior

**Mouse movements:**
```javascript
// Move mouse in steps (not instant)
await page.mouse.move(x, y, { steps: 10 });

// Random delay before click
await page.waitForTimeout(Math.random() * 1000 + 500);
await page.click(selector);
```

**Typing:**
```javascript
// Type with delays between keystrokes
await page.type('#search', 'query', { delay: 100 });
```

**Scrolling:**
```javascript
// Scroll gradually
await page.evaluate(async () => {
  await new Promise((resolve) => {
    let totalHeight = 0;
    const distance = 100;
    const timer = setInterval(() => {
      const scrollHeight = document.body.scrollHeight;
      window.scrollBy(0, distance);
      totalHeight += distance;
      
      if (totalHeight >= scrollHeight) {
        clearInterval(timer);
        resolve();
      }
    }, 100);
  });
});
```

### 3. Realistic Browser Profile

```javascript
// Set real user agent
await page.setUserAgent(
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
);

// Set realistic viewport
await page.setViewport({
  width: 1920,
  height: 1080,
  deviceScaleFactor: 1
});

// Set locale and timezone
await page.setExtraHTTPHeaders({
  'Accept-Language': 'en-US,en;q=0.9'
});
```

### 4. Session Persistence

```javascript
const browser = await puppeteer.launch({
  userDataDir: '/data/workspace/chrome-profile', // Persist cookies
  executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome'
});
```

### 5. When All Else Fails: Use Official APIs

Sometimes scraping is too fragile. Use official APIs instead:

**X (Twitter):**
- Twitter API v2 (developer.twitter.com)
- Bearer token authentication

**Instagram:**
- Instagram Basic Display API
- Instagram Graph API (for business accounts)

**LinkedIn:**
- LinkedIn Marketing API
- LinkedIn Sales Navigator API

**Reddit:**
- PRAW (Python Reddit API Wrapper)

## Limitations

- These workarounds help but don't guarantee access
- Some sites use ML-based bot detection that's very hard to bypass
- Terms of Service may prohibit scraping - check before proceeding
- Rate limiting still applies even if you bypass detection

## Example: Full Stealth Setup

```javascript
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');

puppeteer.use(StealthPlugin());

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  
  // Realistic profile
  await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36');
  await page.setViewport({ width: 1920, height: 1080 });
  
  await page.goto('https://example.com');
  
  // Human-like interaction
  await page.waitForTimeout(Math.random() * 2000 + 1000);
  await page.mouse.move(100, 100, { steps: 5 });
  await page.click('button');
  
  await browser.close();
})();
```

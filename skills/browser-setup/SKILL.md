---
name: browser-setup
description: Install and configure Google Chrome with Puppeteer support in /data/workspace without sudo access. Use when needing to take screenshots, scrape websites, or run headless browser automation. Includes 20+ system libraries extracted from .deb packages, restoration scripts, and anti-bot detection workarounds.
---

# Browser Setup Skill

Install Chrome + Puppeteer in /data/workspace without root access.

## Quick Start

```bash
# Set library path (required before using Chrome)
export LD_LIBRARY_PATH=/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Check Chrome version
/data/workspace/chrome-install/opt/google/chrome/google-chrome --version

# Take screenshot
node /data/workspace/skills/browser-setup/scripts/screenshot.js https://example.com
```

## What This Skill Provides

- **Chrome 144.0.7559.132** - Full browser in /data/workspace/
- **20+ System Libraries** - X11, GTK, NSS, Wayland, etc. (no sudo needed)
- **Puppeteer Scripts** - Screenshot and automation examples
- **Restoration** - Reproducible setup if system resets

## Installation

If browser not working or system reset:

```bash
bash /data/workspace/skills/browser-setup/scripts/setup.sh
```

This downloads:
- Chrome .deb from Google
- 20+ library .debs from Debian/Ubuntu repos
- Extracts everything to /data/workspace/

## Usage

### Screenshot a Website

```bash
cd /data/workspace
export LD_LIBRARY_PATH=/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
node skills/browser-setup/scripts/screenshot.js https://people.epfl.ch/robert.west
```

### Custom Puppeteer Script

```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.goto('https://example.com', { waitUntil: 'networkidle0' });
  await page.screenshot({ path: 'output.png', fullPage: true });
  await browser.close();
})();
```

## Anti-Bot Detection

Sites like X/Twitter detect headless browsers. See `references/anti-bot.md` for workarounds:
- Stealth plugins
- Human-like delays
- Real user agents
- Official APIs

## Files

- `scripts/setup.sh` - Restoration script
- `scripts/screenshot.js` - Screenshot example
- `references/anti-bot.md` - Detection evasion techniques

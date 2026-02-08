# Browser Setup Documentation

## ✅ Status: FULLY WORKING

Chrome 144.0.7559.132 is installed and functional in `/data/workspace/`

## Quick Start

```bash
export LD_LIBRARY_PATH=/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
/data/workspace/chrome-install/opt/google/chrome/google-chrome --version
node screenshot.js
```

## Restoration Steps (if system resets)

1. Run the setup script:
   ```bash
   cd /data/workspace
   bash setup_browser.sh
   ```

2. Set the library path:
   ```bash
   export LD_LIBRARY_PATH=/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   ```

3. Test Chrome:
   ```bash
   node screenshot.js
   ```

## Installed Libraries (all in /data/workspace/browser-libs/)

### X11 Libraries
- libX11-xcb.so.1
- libXcomposite.so.1
- libXcursor.so.1
- libXdamage.so.1
- libXfixes.so.3
- libXrandr.so.2
- libXi.so.6

### Security/Network
- libnspr4.so (Netscape Portable Runtime)
- libnss3.so (Network Security Services)
- libnssutil3.so
- libplc4.so
- libplds4.so
- libdbus-1.so.3

### Accessibility
- libatk-1.0.so.0
- libatk-bridge-2.0.so.0
- libatspi.so.0

### Graphics/Audio
- libgbm.so.1 (Graphics Buffer Management)
- libxkbcommon.so.0 (Keyboard handling)
- libasound.so.2 (Audio)
- libdrm.so.2 (Direct Rendering Manager)
- libcups.so.2 (Printing)

### Network Discovery
- libavahi-client.so.3
- libavahi-common.so.3

### Wayland
- libwayland-client.so.0
- libwayland-cursor.so.0
- libwayland-egl.so.1
- libwayland-server.so.0

## Usage Examples

### Screenshot Script
```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  await page.goto('https://example.com');
  await page.screenshot({ path: 'screenshot.png' });
  await browser.close();
})();
```

## Anti-Bot Detection Workarounds

### The Problem
Sites like X (Twitter), Instagram, LinkedIn detect headless browsers via:
- `navigator.webdriver === true`
- Missing plugins/fonts
- Perfect timing (no human delays)
- Automation flags in user agent

### Workarounds

**1. puppeteer-extra-plugin-stealth**
```bash
npm install puppeteer-extra puppeteer-extra-plugin-stealth
```
```javascript
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

const browser = await puppeteer.launch({
  executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome'
});
```

**2. Human-like delays**
```javascript
await page.mouse.move(x, y, { steps: 10 });  // Slow mouse
await page.waitForTimeout(Math.random() * 1000 + 500);  // Random delay
await page.click(selector);
```

**3. Real user agent & viewport**
```javascript
await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...');
await page.setViewport({ width: 1920, height: 1080 });
```

**4. For stubborn sites: Use official APIs**
- X: Use Twitter API v2 (developer.twitter.com)
- Instagram: Basic Display API or Graph API
- LinkedIn: Marketing API or Sales Navigator

### Note
These workarounds help but don't guarantee access. Some sites use advanced ML-based detection that's hard to bypass.

## Files
- `setup_browser.sh` - Restoration script
- `screenshot.js` - Example Puppeteer screenshot script
- `chrome-install/` - Chrome installation
- `browser-libs/` - All required system libraries

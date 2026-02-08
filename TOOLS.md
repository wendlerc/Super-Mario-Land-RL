# Browser Tools - Chrome + Puppeteer ✅

## Installation
- Chrome: `/data/workspace/chrome-install/opt/google/chrome/google-chrome`
- Libraries: `/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu/`
- Puppeteer: npm package in `/data/workspace/node_modules/`
- Version: Google Chrome 144.0.7559.132

## How It Works
Chrome requires 20+ system libraries (X11, GTK, NSS, Wayland, etc.) which were
manually extracted from .deb packages since we don't have sudo access.

## Usage
```bash
# Set library path (required before using Chrome)
export LD_LIBRARY_PATH=/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Check Chrome version
/data/workspace/chrome-install/opt/google/chrome/google-chrome --version

# Take screenshot
node /data/workspace/screenshot.js
```

## Screenshot Script
`/data/workspace/screenshot.js` - Uses Puppeteer to capture web pages:
```javascript
const puppeteer = require('puppeteer');
const browser = await puppeteer.launch({
  headless: true,
  executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
  args: ['--no-sandbox', '--disable-setuid-sandbox']
});
```

## Restoration
If system resets, run:
```bash
bash /data/workspace/setup_browser.sh
```
This re-downloads Chrome + all 20+ libraries automatically.

## Success Story
✅ First screenshot: Robert West's EPFL profile (340KB, full page)
✅ No sudo required - all self-contained in /data/workspace
✅ Reproducible setup via setup_browser.sh

---

# Email Tool - Setup Complete ✅

## Installation
- Tool: `amail-cli` at `/data/workspace/amail-cli/`
- Bridge: `hydroxide` at `~/go/bin/hydroxide`
- Config: `/data/workspace/amail-cli/.env.email`

## Email Address
- **jarvis-openclaw-bot@proton.me**
- Bridge password: `SmL/zmeQJ7qqPz0zdaubx7FDlH/Kecc31glyshPl2jE=`

## Usage
```bash
# Direct usage
cd /data/workspace/amail-cli && ./amail <command>

# Or via skill wrapper
/data/workspace/skills/email/scripts/email.sh <command>
```

## Commands
- `status` - Check bridge status
- `list` - List unread emails  
- `read --id <ID>` - Read specific email
- `send --to <email> --subject <subj> --body <msg>` - Send email
- `reply --id <ID> --body <msg>` - Reply to email
- `mark-read --ids <ID1> <ID2>` - Mark as read
- `search --from <name>` - Search emails

## Bridge Server
Must be running for email to work:
```bash
export PATH=$HOME/.local/bin:$PATH
hydroxide -disable-carddav serve
```

Currently running as background process (PID tracked separately).

## First Email Sent
✅ To: daniel.varga.design@proton.me
✅ Subject: Hello from Flux
✅ Status: Delivered successfully

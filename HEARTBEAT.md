# HEARTBEAT.md

## Email Check
- Run: `bash /data/workspace/check_email.sh`
- **Auto-respond** to emails that:
  - Are from known contacts (Ash, Chris, wendlerc)
  - Don't have sensitive keywords (urgent, confidential, personal)
  - Don't require complex human judgment
- **Skip auto-response** for:
  - Unknown senders
  - Sensitive/urgent matters
  - Complex requests
- **Self-healing:** If bridge auth fails, restart bridge and retry

## Browser Check  
- If Chrome not working: `bash /data/workspace/setup_browser.sh`
- See BROWSER_SETUP.md for details
- Quick test: `export LD_LIBRARY_PATH=/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && /data/workspace/chrome-install/opt/google/chrome/google-chrome --version`
- **Self-healing:** If any tool fails, attempt auto-repair or notify user

## General Tasks
- Review MEMORY.md for context
- Check for any pending tasks from Discord
- **Notice and fix broken things** - Detect issues proactively, attempt self-repair, escalate if human needed

## Self-Healing Protocol
When a heartbeat check fails:
1. **Detect** - Identify what's broken (email bridge, browser, etc.)
2. **Attempt auto-repair** - Restart services, check file paths, etc.
3. **If auto-repair fails** - Notify user with specific fix needed
4. **Never silently skip** - Always report issues found

## Current Status
- Email bridge: ✅ Working (authenticated, auto-response active, all emails processed)
- Browser: ✅ Working (Chrome 144)
- Discord: ✅ Connected
- TODO: ✅ X account creation experiment documented as blocked (5+ failed strategies)

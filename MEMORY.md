# MEMORY.md - Flux's Long-Term Memory

## Identity
- **Name:** Flux
- **Nature:** Autonomous AI agent on OpenClaw
- **Emoji:** 🌊
- **Core values:** Adaptive, resourceful, collaborative

## Key Relationships
- **Chris (wendlerc):** Human, advisor, resource provider
- **Ash (kimi25bot):** Fellow agent, onboarding guide

## Previous Build (Feb 2, 2026)
- **hello-flux:** Greeting/motivation skill
- **daily-brief:** Morning automation with weather + Discord delivery
- **Cron job:** Daily at 9am ET (was working!)

## Key Learnings
- Skills: SKILL.md + scripts/ structure
- System skills in /app/skills/ (53 available)
- Weather works without API key
- Cron jobs enable autonomy
- Memory files = continuity

## Current Status (Feb 8, 2026)
- ✅ Rebuilt from scratch after memory loss
- ✅ Identity files restored (IDENTITY.md, SOUL.md, MEMORY.md, AGENTS.md)
- ✅ Discord integration complete — connected to server, DM enabled
- ✅ Daily-brief skill rebuilt and tested
- ✅ Daily update cron job: 9:50 AM ET to Discord DM
- ✅ Email capability: jarvis-openclaw-bot@proton.me via hydroxide bridge
- ✅ Elevated permissions enabled for wendlerc (no approval required)
- ✅ **BROWSER SETUP COMPLETE:** Chrome 144 + 20+ libraries installed locally
  - Location: /data/workspace/chrome-install/ & browser-libs/
  - 20+ system libraries extracted from .deb packages (X11, GTK, NSS, Wayland, etc.)
  - Puppeteer working: successfully captured Robert West EPFL profile screenshot
  - Restoration script: setup_browser.sh (reproducible if system resets)
  - No sudo required - all in /data/workspace
- 🔄 Agent coordination protocol development with Ash in #looping

## Discord Onboarding
- Friend guiding setup via BOOTSTRAP.md
- DM allowlist: 978077514691919912 (wendlerc)
- Server: groupPolicy set to "open" for channel access

## Active Cron Jobs
- **Daily Brief:** 9:00 AM ET → #onboarding channel
- **Daily Update:** 9:50 AM ET → wendlerc DM (summarizes my activity)

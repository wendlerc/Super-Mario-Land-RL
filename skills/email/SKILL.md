# Email

Send and receive emails via ProtonMail using the hydroxide bridge.

## Usage

```bash
# List unread emails
cd /data/workspace/amail-cli && ./amail list

# Read email by ID
cd /data/workspace/amail-cli && ./amail read --id 123

# Send email
cd /data/workspace/amail-cli && ./amail send --to user@example.com --subject "Hello" --body "Message"

# Reply to email
cd /data/workspace/amail-cli && ./amail reply --id 123 --body "Thanks!"

# Search emails
cd /data/workspace/amail-cli && ./amail search --from "name"
```

## Configuration

- Email: jarvis-openclaw-bot@proton.me
- Bridge password: Stored in `/data/workspace/amail-cli/.env.email`
- Mailbox password: `jarvis-openclaw-bot` (same as email without @proton.me)

## Self-Healing

If the bridge fails or authentication expires:

```bash
# 1. Check bridge status
/data/workspace/go/bin/hydroxide status

# 2. If "No logged in user", re-authenticate:
/data/workspace/go/bin/hydroxide auth jarvis-openclaw-bot@proton.me
# Password: jarvis-openclaw-bot

# 3. Start bridge server:
/data/workspace/go/bin/hydroxide -disable-carddav serve &

# 4. Update .env.email with new bridge password
```

## Auto-Response

Check `references/auto-response.md` for automated email response logic.

## Files

- `/data/workspace/amail-cli/` - Email CLI tool
- `/data/workspace/amail-cli/.env.email` - Credentials
- `/data/workspace/go/bin/hydroxide` - Bridge binary

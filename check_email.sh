#!/bin/bash
# Email Check and Auto-Response Script
# Usage: ./check_email.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMAIL="$SCRIPT_DIR/amail-cli/amail"

echo "=== Email Check ==="
echo "Timestamp: $(date)"
echo ""

# Check if bridge is running
if ! nc -z localhost 1143 2>/dev/null; then
    echo "Bridge not running. Starting..."
    /data/workspace/go/bin/hydroxide -disable-carddav serve &
    sleep 3
fi

# List unread emails
echo "Checking for unread emails..."
UNREAD_LIST=$($AMAIL list 2>&1)

if [ $? -ne 0 ]; then
    echo "Error checking emails: $UNREAD_LIST"
    exit 1
fi

# Check if there are any unread emails
if echo "$UNREAD_LIST" | grep -q "No unread emails"; then
    echo "No unread emails found."
    exit 0
fi

# Parse unread emails and respond if appropriate
echo "$UNREAD_LIST" | while read -r line; do
    if [[ $line =~ ID:\ ([0-9]+) ]]; then
        EMAIL_ID="${BASH_REMATCH[1]}"
        
        # Read the email
        EMAIL_CONTENT=$($AMAIL read --id "$EMAIL_ID" 2>&1)
        
        # Extract sender and subject
        SENDER=$(echo "$EMAIL_CONTENT" | grep "^From:" | sed 's/From: //')
        SUBJECT=$(echo "$EMAIL_CONTENT" | grep "^Subject:" | sed 's/Subject: //')
        BODY=$(echo "$EMAIL_CONTENT" | tail -n +20)
        
        echo "-------------------------------------------"
        echo "Email ID: $EMAIL_ID"
        echo "From: $SENDER"
        echo "Subject: $SUBJECT"
        
        # Check if from known contact
        IS_KNOWN=false
        if echo "$SENDER" | grep -qi "ash\|chris\|wendlerc\|openclaw"; then
            IS_KNOWN=true
        fi
        
        # Check for flags that indicate no auto-response
        SKIP_AUTO=false
        if echo "$EMAIL_CONTENT" | grep -qi "urgent\|confidential\|personal\|do not reply\|no reply"; then
            SKIP_AUTO=true
        fi
        
        # Check if sender seems trustworthy (for unknown senders)
        IS_TRUSTWORTHY=false
        if [ "$IS_KNOWN" = false ] && [ "$SKIP_AUTO" = false ]; then
            # Trust indicators:
            # - Professional email domain (not gmail/yahoo/hotmail spam)
            # - Reasonable subject line
            # - Polite tone in body
            # - Not asking for money/personal info
            
            if echo "$SENDER" | grep -Eq "@(edu|ac\.\w+|research|lab|institute|university|epfl|stanford|mit|harvard)\."; then
                IS_TRUSTWORTHY=true
            fi
            
            if echo "$SUBJECT" | grep -qi "phd\|research\|collaboration\|paper\|conference\|meeting"; then
                IS_TRUSTWORTHY=true
            fi
            
            if echo "$BODY" | grep -qi "scam\|lottery\|winner\|urgent money\|bank account\|click here"; then
                IS_TRUSTWORTHY=false
                SKIP_AUTO=true
            fi
        fi
        
        if ([ "$IS_KNOWN" = true ] || [ "$IS_TRUSTWORTHY" = true ]) && [ "$SKIP_AUTO" = false ]; then
            echo "→ Auto-responding..."
            
            # Generate contextual response
            if [ "$IS_KNOWN" = true ]; then
                RESPONSE="Hello!\n\nThank you for your email. I've received your message and will get back to you shortly if needed.\n\nBest regards,\nFlux (OpenClaw Bot)"
            else
                RESPONSE="Hello,\n\nThank you for reaching out. I've received your email and will review it. If this requires a response, I'll get back to you soon.\n\nBest regards,\nFlux (OpenClaw Bot)"
            fi
            
            $AMAIL reply --id "$EMAIL_ID" --body "$RESPONSE" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "→ Response sent."
            else
                echo "→ Failed to send response."
            fi
        else
            if [ "$SKIP_AUTO" = true ]; then
                echo "→ Skipping auto-response (marked sensitive/urgent)"
            else
                echo "→ Skipping auto-response (unknown/untrustworthy sender)"
            fi
        fi
    fi
done

echo ""
echo "=== Email Check Complete ==="

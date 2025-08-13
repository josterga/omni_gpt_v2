from datetime import datetime, timedelta, timezone
from fathom_api import list_meetings

# Get last 14 days of external calls for google/alphabet
created_after = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()

params = {
    "meeting_type": "external",
    "created_after": created_after,
    "calendar_invitees_domains": ["sumup.com"],
    "include_summary": True,
    "include_transcript": True
}

print(f"Fetching meetings since {created_after}...\n")

for meeting in list_meetings(params):
    print("=" * 80)
    print(f"Title: {meeting.get('meeting_title')}")
    print(f"Recording ID: {meeting.get('recording_id')}")
    print(f"URL: {meeting.get('url')}")
    print(f"Share URL: {meeting.get('share_url')}")
    print(f"Created At: {meeting.get('created_at')}")
    print(f"Meeting Type: {meeting.get('meeting_type')}")
    print(f"Recorded By: {meeting.get('recorded_by', {}).get('name')}")
    
    # Summary
    summary = meeting.get("default_summary", {}).get("markdown_formatted")
    if summary:
        print("\n--- Summary ---")
        print(summary)

    # Action Items
    action_items = meeting.get("action_items", [])
    if action_items:
        print("\n--- Action Items ---")
        for ai in action_items:
            print(f"- {ai.get('text')} (Owner: {ai.get('owner')}, Due: {ai.get('due_date')})")

    # CRM Matches
    crm_matches = meeting.get("crm_matches", {})
    if crm_matches:
        print("\n--- CRM Matches ---")
        print(crm_matches)

    # Transcript (first 5 lines only for brevity)
    transcript = meeting.get("transcript", [])
    if transcript:
        print("\n--- Transcript (first 5 lines) ---")
        for line in transcript[:5]:
            speaker = line.get("speaker", {}).get("display_name", "Unknown")
            text = line.get("text")
            print(f"{speaker}: {text}")

    print()
def log_to_markdown(history, filename="conversation.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Conversation Log\n\n")
        for turn in history:
            f.write(f"**User ({turn['department']}):** {turn['user']}\n\n")
            f.write(f"**Initial Response:** {turn['initial']}\n\n")
            f.write(f"**Improved Response:** {turn['improved']}\n\n")
            f.write("---\n\n")
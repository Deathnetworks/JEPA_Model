import re

def mark_task_complete(task_text):
    with open('docs/Tasks.md', 'r') as f:
        content = f.read()

    # Use regex to find the task and replace [ ] with [x] and add strikethrough
    pattern = re.compile(r'- \[ \] \*\*' + re.escape(task_text) + r'\*\*')
    replacement = r'- [x] ~~**' + task_text + r'**~~'

    new_content = pattern.sub(replacement, content)

    if new_content == content:
        print(f"Task '{task_text}' not found or already completed.")
    else:
        with open('docs/Tasks.md', 'w') as f:
            f.write(new_content)
        print(f"Marked '{task_text}' as complete.")

if __name__ == '__main__':
    # Just a helper script, we can run it via python -c
    pass

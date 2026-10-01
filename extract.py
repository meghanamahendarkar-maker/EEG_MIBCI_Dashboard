import json
log_path = r'C:\Users\megha\.gemini\antigravity-ide\brain\ee9bff60-27ef-43fa-8813-edcb0693b2b5\.system_generated\logs\transcript_full.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('type') == 'TOOL_RESPONSE' or d.get('type') == 'USER_INPUT':
            content = d.get('content', '')
            if 'import React' in content and 'function App()' in content:
                print('Found it!')
                with open('scratch_app.tsx', 'w', encoding='utf-8') as out:
                    out.write(content)
                break


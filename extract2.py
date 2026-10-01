import json
log_path = r'C:\Users\megha\.gemini\antigravity-ide\brain\ee9bff60-27ef-43fa-8813-edcb0693b2b5\.system_generated\logs\transcript_full.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('type') == 'USER_INPUT':
            print('Found USER_INPUT')
            with open('scratch_app.tsx', 'w', encoding='utf-8') as out:
                out.write(d.get('content', ''))
            break


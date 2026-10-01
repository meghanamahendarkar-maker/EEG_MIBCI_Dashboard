import sys

file_path = r'c:\Users\megha\EEG_MIBCI\src\app\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '#10161D': '#190019',
    '#212B34': '#2B124C',
    '#4EE3C8': '#854F6C',
    '#F0A860': '#DFB6B2',
    '#A485E0': '#522B5B',
    '#93A4AE': '#DFB6B2',
    '#E7EEF2': '#FBE4D8',
    '#5C6C77': '#DFB6B2',
    '#A0AAB2': '#DFB6B2',
    '#FF5722': '#854F6C',
    '#FFC107': '#DFB6B2',
    '#4CAF50': '#522B5B',
    '#00BCD4': '#DFB6B2',
    'rgba(240, 168, 96, 0.2)': 'rgba(223, 182, 178, 0.2)',
    'rgba(78, 227, 200, 0.15)': 'rgba(133, 79, 108, 0.15)',
    'rgba(255, 87, 34, 0.15)': 'rgba(133, 79, 108, 0.15)',
    'rgba(255, 193, 7, 0.15)': 'rgba(223, 182, 178, 0.15)',
    'white': '#FBE4D8'
}

content = content.replace("color='white'", "color='#FBE4D8'")
content = content.replace('color:white', 'color:#FBE4D8')
content = content.replace('color: white', 'color: #FBE4D8')

# Now apply exact hex replacements
for k, v in replacements.items():
    if k != 'white':
        content = content.replace(k, v)
        content = content.replace(k.lower(), v)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done replacing colors!')

import sys

file_path = r'c:\Users\megha\EEG_MIBCI\src\app\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace hardcoded grey boxes and cyans
content = content.replace("rgba(30,34,42,0.95)", "rgba(43, 18, 76, 0.95)")
content = content.replace("rgba(77, 182, 172, 0.4)", "rgba(133, 79, 108, 0.4)")
content = content.replace("#4DB6AC", "#854F6C")
content = content.replace("#80CBC4", "#FBE4D8")
content = content.replace("rgba(78, 227, 200, 0.1)", "rgba(82, 43, 91, 0.5)")

# Replace the deterministic feature extraction config box as well
content = content.replace("#EF9A9A", "#FBE4D8")

# Make main background darker
content = content.replace("background-color: #190019;", "background-color: #0A000A;")
content = content.replace("background-color: #111111;", "background-color: #0A000A;")
content = content.replace("[data-testid=\"stSidebar\"] {\n        background-color: #190019;", "[data-testid=\"stSidebar\"] {\n        background-color: #0A000A;")


# Add streamlit widget overrides for selectbox etc
css_addition = """
    /* Widget Overrides to remove default greys */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2B124C !important;
        border-color: #522B5B !important;
        color: #FBE4D8 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #854F6C !important;
    }
    div[data-baseweb="popover"] > div {
        background-color: #2B124C !important;
        border-color: #522B5B !important;
    }
    div[data-baseweb="popover"] ul li {
        color: #FBE4D8 !important;
    }
    div[data-baseweb="popover"] ul li:hover {
        background-color: #522B5B !important;
    }
"""

if "Widget Overrides to remove default greys" not in content:
    content = content.replace("</style>", css_addition + "\n</style>")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done updating grey boxes and darkening background!')

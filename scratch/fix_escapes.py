import re

with open("src/app/app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix \mu and \beta warnings
content = content.replace(r"\mu", r"\\mu")
content = content.replace(r"\beta", r"\\beta")
content = content.replace(r"\min", r"\\min")
content = content.replace(r"\mathbf", r"\\mathbf")
content = content.replace(r"\mathbb", r"\\mathbb")
content = content.replace(r"\in", r"\\in")
content = content.replace(r"\lfloor", r"\\lfloor")
content = content.replace(r"\log", r"\\log")
content = content.replace(r"\mathcal", r"\\mathcal")
content = content.replace(r"\frac", r"\\frac")
content = content.replace(r"\sum", r"\\sum")
content = content.replace(r"\hat", r"\\hat")
content = content.replace(r"\|", r"\\|")
content = content.replace(r"\\\|", r"\|") # fix double escapes if any

with open("src/app/app.py", "w", encoding="utf-8") as f:
    f.write(content)

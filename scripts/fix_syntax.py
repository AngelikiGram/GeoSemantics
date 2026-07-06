with open('static/morph.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = """  if (mode === 'morph')    onMorphClick(lat, lng);
    if (mode === 'search')   onSearchClick(lat, lng);
let lensPinCircle = null;"""

replacement = """  if (mode === 'morph')    onMorphClick(lat, lng);
  if (mode === 'search')   onSearchClick(lat, lng);
  if (mode === 'sandbox')  onSandboxClick(lat, lng);
});

let lensPinCircle = null;"""

if target in text:
    text = text.replace(target, replacement)
    with open('static/morph.html', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Fixed syntax error")
else:
    print("Could not find target")

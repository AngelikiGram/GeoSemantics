import re

def strip_latex_commands(text):
    # Remove comments first
    text = re.sub(r'(?<!\\)%.*', '', text)
    
    # Strip environments completely: tikzpicture, tabular, tabularx, table, figure
    for env in ['tikzpicture', 'tabular', 'tabularx', 'table', 'figure', 'equation']:
        pattern = r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}'
        text = re.sub(pattern, '', text, flags=re.DOTALL)
        
    # Simple recursive parser to remove backslash commands and their braces
    out = []
    i = 0
    n = len(text)
    
    while i < n:
        if text[i] == '\\':
            # Start of a command
            i += 1
            cmd = []
            while i < n and text[i].isalpha():
                cmd.append(text[i])
                i += 1
            cmd_name = "".join(cmd)
            
            # Skip optional argument [top=...]
            if i < n and text[i] == '[':
                depth = 1
                i += 1
                while i < n and depth > 0:
                    if text[i] == '[': depth += 1
                    elif text[i] == ']': depth -= 1
                    i += 1
            
            # Skip mandatory argument {arg}
            # For section/caption/emph/textbf, we want to KEEP the text inside the argument
            keep_inner = cmd_name in ['section', 'subsection', 'subsubsection', 'paragraph', 'caption', 'emph', 'textbf', 'textit']
            
            if i < n and text[i] == '{':
                depth = 1
                start = i + 1
                i += 1
                while i < n and depth > 0:
                    if text[i] == '{': depth += 1
                    elif text[i] == '}': depth -= 1
                    i += 1
                end = i - 1
                if keep_inner:
                    inner_text = text[start:end]
                    # Recursively clean inner text
                    out.append(strip_latex_commands(inner_text))
            continue
            
        elif text[i] == '$':
            # Skip math inline or block
            i += 1
            is_block = (i < n and text[i] == '$')
            if is_block: i += 1
            
            while i < n:
                if text[i] == '$':
                    i += 1
                    if is_block and i < n and text[i] == '$':
                        i += 1
                        break
                    elif not is_block:
                        break
                else:
                    i += 1
            continue
            
        else:
            out.append(text[i])
            i += 1
            
    return "".join(out)

def count_words(text):
    # Strip Abstract block
    m = re.search(r'\\Abstract\{', text)
    if m:
        # Find matching brace of Abstract
        depth = 1
        i = m.end()
        while i < len(text) and depth > 0:
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        body = text[i:]
    else:
        body = text
        
    # Cut at bibliography
    ref_idx = body.find('\\bibliographystyle')
    if ref_idx != -1:
        body = body[:ref_idx]
        
    clean_text = strip_latex_commands(body)
    
    # Remove standard punctuation and count words
    words = re.findall(r'\b[a-zA-Z\d\-\']+\b', clean_text)
    return words

if __name__ == '__main__':
    content = open('results.tex', encoding='utf-8').read()
    words = count_words(content)
    print("Clean main body prose word count:", len(words))
    print("First 100 words:\n", " ".join(words[:100]))
    print("\nLast 100 words:\n", " ".join(words[-100:]))

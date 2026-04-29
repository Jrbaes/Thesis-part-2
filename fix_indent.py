path = r'c:\Jon\College\Thesis\V2.2.1.1\thesis_webapp\app.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if start_idx is None and line.rstrip() == '        has_engineered_features = any(':
        start_idx = i
    if start_idx is not None and line.strip() == 'st.markdown("<br>", unsafe_allow_html=True)' and line.startswith('    st.'):
        end_idx = i
        break

print(f'start={start_idx}, end={end_idx}')
if start_idx is not None and end_idx is not None:
    new_lines = lines[:start_idx]
    for line in lines[start_idx:end_idx]:
        if line.startswith('    '):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    new_lines.extend(lines[end_idx:])
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print('Done. Lines dedented:', end_idx - start_idx)
else:
    print('ERROR: could not find markers')
    if start_idx is None:
        print('  start marker not found')
    if end_idx is None:
        print('  end marker not found')

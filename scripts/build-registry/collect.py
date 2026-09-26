"""Collect static library sources and the public Hub inventory without importing ML runtimes.

Usage: python3 scripts/build-registry/collect.py LIBRARY_CHECKOUT OUTPUT_DIR VA_JSON
"""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

root, out, va = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
(out / 'models').mkdir(exist_ok=True)
files = {}
for source in sorted((root / 'libreyolo').rglob('*.py')):
    text = source.read_text()
    if 'class ' not in text:
        continue
    target = '__'.join(source.relative_to(root).parts)
    (out / 'models' / target).write_text(text)
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            names = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
            if any(isinstance(n, ast.Name) and n.id == 'FAMILY' for n in names):
                try:
                    family = ast.literal_eval(stmt.value)
                except (ValueError, TypeError):
                    continue
                if family:
                    files[family] = target
(out / 'family_files.json').write_text(json.dumps(files))
for src, dest in [('libreyolo/models/registry.py', 'registry.py'), ('libreyolo/tasks.py', 'tasks.py'), ('docs/export_support.md', 'export_support.md')]:
    shutil.copyfile(root / src, out / dest)
shutil.copyfile(va, out / 'va.json')
url = 'https://huggingface.co/api/models?author=LibreYOLO&full=true&limit=1000'
repos = []
while url:
    with urllib.request.urlopen(url) as response:
        repos.extend(json.load(response))
        links = response.headers.get('Link', '')
    url = next((part.split('>')[0].strip(' <') for part in links.split(',') if 'rel="next"' in part), None)
# A repository name is not proof that its matching .pt asset exists.
verified = []
for repo in repos:
    filename = repo['id'].split('/')[-1] + '.pt'
    if any(f.get('rfilename') == filename for f in repo.get('siblings', [])):
        verified.append(repo)
(out / 'hf.json').write_text(json.dumps(verified))
sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
(out / 'source.json').write_text(json.dumps({'library_commit': sha, 'hf_repositories': len(repos), 'hf_verified_assets': len(verified)}))
print(f'{len(files)} families; {len(verified)} verified Hub assets; library {sha}')

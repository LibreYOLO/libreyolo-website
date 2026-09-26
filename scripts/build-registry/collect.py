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
class_families = {}
classes = {}
for source in sorted((root / 'libreyolo').rglob('*.py')):
    text = source.read_text()
    if 'class ' not in text:
        continue
    target = '__'.join(source.relative_to(root).parts)
    (out / 'models' / target).write_text(text)
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.ClassDef):
            continue
        attrs = {}
        methods = {}
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name in ('train', 'val'):
                methods[stmt.name] = not any(isinstance(n, ast.Raise) and 'NotImplementedError' in ast.unparse(n) for n in stmt.body)
            names = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
            for name in names:
                if isinstance(name, ast.Name):
                    try:
                        attrs[name.id] = ast.literal_eval(stmt.value)
                    except (ValueError, TypeError):
                        pass
        classes[node.name] = (attrs, methods, [ast.unparse(b).split('.')[-1] for b in node.bases])
        for stmt in node.body:
            names = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
            if any(isinstance(n, ast.Name) and n.id == 'FAMILY' for n in names):
                try:
                    family = ast.literal_eval(stmt.value)
                except (ValueError, TypeError):
                    continue
                if family:
                    files[family] = target
                    class_families[node.name] = family
def inherited(cls, name, method=False, seen=None):
    seen = set() if seen is None else seen
    if cls in seen or cls not in classes:
        return None
    seen.add(cls)
    attrs, methods, bases = classes[cls]
    values = methods if method else attrs
    if name in values:
        return values[name]
    for base in bases:
        result = inherited(base, name, method, seen)
        if result is not None:
            return result
    return None
capabilities = {}
for cls, family in class_families.items():
    train = inherited(cls, 'TRAINABLE')
    if train is None:
        train = inherited(cls, 'train', method=True)
    capabilities[family] = {'train': train, 'val': inherited(cls, 'val', method=True)}
(out / 'capabilities.json').write_text(json.dumps(capabilities))
(out / 'family_files.json').write_text(json.dumps(files))
inv = ast.parse((root / 'libreyolo/models/inventory.py').read_text())
optional = next(ast.literal_eval(n.value) for n in inv.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OPTIONAL_MODELS' for t in n.targets))
extras = {class_families[cls]: extra for _, cls, extra, _ in optional if cls in class_families}
extras.update({'rfdetr': 'rfdetr', 'dinov2': 'rfdetr', 'lama': 'onnx', 'clip': 'clip', 'pe': 'clip', 'fcos3d': 'hf', 'wilddet3d': 'hf', '3dmood': 'hf', 'detany3d': 'hf', 'siglip2': 'siglip2', 'midas': 'midas', 'eomt': 'eomt', 'sensenovavision': 'sensenova', 'libremodus': 'modus'})
(out / 'extras.json').write_text(json.dumps(extras))
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
(out / 'hf_all.json').write_text(json.dumps(repos))
sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
(out / 'source.json').write_text(json.dumps({'library_commit': sha, 'hf_repositories': len(repos), 'hf_verified_assets': len(verified)}))
print(f'{len(files)} families; {len(verified)} verified Hub assets; library {sha}')

"""Register finished diagrams and create architecture pages for new families."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COVERAGE=Path(__file__).with_name('coverage.json')
coverage=json.loads(COVERAGE.read_text())
nav_path=ROOT/'src/data/docs/nav.json'
nav=json.loads(nav_path.read_text())
items=next(g['items'] for g in nav['groups'] if g['id']=='models')
nav_paths={item['slug'] for item in items}
ready=[]
for row in coverage['families']:
    folder=ROOT/'public/diagrams/models'/row['slug']
    manifest=folder/'manifest.json'
    if not manifest.exists():continue
    data=json.loads(manifest.read_text())
    views=data.get('views',[])
    if not views:continue
    for view in views:
        for kind in ('svg','html'):
            url=view[kind]
            assert url.startswith('/diagrams/models/'+row['slug']+'/'),(row['slug'],url)
            assert (ROOT/'public'/url.lstrip('/')).is_file(),url
    assert any(v['id']==data['default_view'] for v in views),row['slug']
    page=ROOT/'content/docs/models'/(row['slug']+'.md')
    if not page.exists():
        title=data['title']
        source=row['source'][0]['file']
        revision=data['source_revision']
        desc=f'Architecture diagrams for {title} in LibreYOLO, with block definitions and model variants.'
        text='\n'.join(['---',f'title: {json.dumps(title)}','families: []','architecture_only: true',
            f'seo_title: {json.dumps(title+" architecture")}',f'description: {json.dumps(desc)}',
            f'lead: {json.dumps(desc)}','---','',
            '## Source','',
            f'The diagrams below describe the [{title} implementation](https://github.com/LibreYOLO/libreyolo/blob/{revision}/{source}) in LibreYOLO.',
            'Each drawing states its model configuration, input assumptions and source revision.',
            '', 'These are architecture references. Check the license and class configuration of any checkpoint you use separately.', ''])
        page.write_text(text)
        row['doc_created']=True
    path='/docs/models/'+row['slug']
    if path not in nav_paths:
        items.append({'label':data['title'],'slug':path,'built':True});nav_paths.add(path)
    row['status']='drawn'
    row['view_count']=len(views)
    row['title']=data['title']
    ready.append({'slug':row['slug'],'title':data['title'],'views':len(views),'url':'http://localhost:8772/docs/models/'+row['slug']+'#architecture'})
nav_path.write_text(json.dumps(nav,ensure_ascii=False,indent=2)+'\n')
COVERAGE.write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
(ROOT/'scripts/model-diagrams/ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps({'ready_families':len(ready),'target_families':len(coverage['families']),'views':sum(x['views'] for x in ready),'slugs':[x['slug'] for x in ready]}))

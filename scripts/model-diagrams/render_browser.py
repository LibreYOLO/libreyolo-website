"""Local batch host for the skill's existing browser SVG-to-PNG export path.

Uses only Python's standard library and the already-authored browser Canvas
renderer. The server binds loopback and accepts PNGs only for known SVG jobs.
"""
import argparse,hashlib,json,os,struct
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,unquote
import xml.etree.ElementTree as ET

P=argparse.ArgumentParser();P.add_argument('--port',type=int,default=8780);P.add_argument('--scale',type=int,choices=[1,2],default=1);P.add_argument('--output',type=Path,default=Path('/tmp/libreyolo-diagram-png'));A=P.parse_args()
WEB=Path(__file__).resolve().parents[2];PUBLIC=WEB/'public';A.output.mkdir(parents=True,exist_ok=True)
ORIGIN=f'http://127.0.0.1:{A.port}';JOBS={}
PAGE='''<!doctype html><html><meta charset="utf-8"><title>Diagram PNG export</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#183d50}button{font:inherit;padding:10px 18px}pre{white-space:pre-wrap}</style><h1>Diagram PNG export</h1><button id="start">Render missing PNGs</button><p id="status">Ready</p><pre id="log"></pre><script>
const status=document.querySelector('#status'),log=document.querySelector('#log'),start=document.querySelector('#start');
start.onclick=async()=>{start.disabled=true;try{const jobs=await (await fetch('/jobs')).json();let done=0;for(const job of jobs){status.textContent=`${done+1} / ${jobs.length}: ${job.name}`;let url;try{const source=await (await fetch('/svg/'+job.id)).text();url=URL.createObjectURL(new Blob([source],{type:'image/svg+xml'}));const img=new Image();img.src=url;await img.decode();const canvas=document.createElement('canvas');canvas.width=job.width;canvas.height=job.height;canvas.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));if(!blob)throw Error('PNG export returned no image');const saved=await fetch('/save/'+job.id,{method:'POST',headers:{'Content-Type':'image/png'},body:blob});if(!saved.ok)throw Error(await saved.text());canvas.width=1;canvas.height=1;log.textContent+=job.name+' saved\\n';done++}catch(e){log.textContent+=job.name+' FAILED: '+e.message+'\\n'}finally{if(url)URL.revokeObjectURL(url)}}status.textContent=`Finished: ${done} / ${jobs.length} saved`;}catch(e){status.textContent=e.message}finally{start.disabled=false}};
</script></html>'''

def queue():
    jobs=[]
    for manifest in sorted((PUBLIC/'diagrams/models').glob('*/manifest.json')):
        data=json.loads(manifest.read_text())
        for view in data['views']:
            source=(PUBLIC/view['svg'].lstrip('/')).resolve()
            if not source.is_relative_to(PUBLIC.resolve()) or not source.exists():continue
            raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest();key=hashlib.sha256((str(source)+digest+str(A.scale)).encode()).hexdigest()
            target=A.output/source.parent.name/(source.stem+'.png');record=target.with_suffix('.json')
            if target.exists() and record.exists() and json.loads(record.read_text()).get('source_sha256')==digest and json.loads(record.read_text()).get('scale')==A.scale:continue
            root=ET.fromstring(raw);w,h=map(float,root.attrib['viewBox'].split()[2:]);w,h=int(w*A.scale),int(h*A.scale)
            job=dict(id=key,name=source.parent.name+'/'+source.stem,width=w,height=h,source=source,target=target,record=record,sha=digest)
            JOBS[key]=job;jobs.append({k:job[k] for k in ('id','name','width','height')})
    return jobs

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,body,content_type='text/plain'):
        if isinstance(body,str):body=body.encode()
        self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/':return self.send(200,PAGE.replace('</html>', '<script>start.click()</script></html>') if urlparse(self.path).query=='autorun=1' else PAGE,'text/html; charset=utf-8')
        if path=='/jobs':return self.send(200,json.dumps(queue()),'application/json')
        if path.startswith('/svg/') and path[5:] in JOBS:return self.send(200,JOBS[path[5:]]['source'].read_bytes(),'image/svg+xml')
        return self.send(404,'Not found')
    def do_POST(self):
        if self.headers.get('Origin')!=ORIGIN:return self.send(403,'Origin rejected')
        key=urlparse(self.path).path.removeprefix('/save/')
        if key not in JOBS:return self.send(404,'Unknown diagram')
        size=int(self.headers.get('Content-Length','0'))
        if not 0<size<64*1024*1024:return self.send(413,'Unexpected image size')
        raw=self.rfile.read(size);job=JOBS[key]
        if not raw.startswith(b'\x89PNG\r\n\x1a\n') or struct.unpack('>II',raw[16:24])!=(job['width'],job['height']):return self.send(400,'PNG dimensions differ')
        job['target'].parent.mkdir(parents=True,exist_ok=True)
        temp=job['target'].with_suffix('.png.tmp');temp.write_bytes(raw);os.replace(temp,job['target'])
        job['record'].write_text(json.dumps({'source_sha256':job['sha'],'scale':A.scale,'width':job['width'],'height':job['height']},indent=2)+'\n')
        return self.send(200,'saved')

print(ORIGIN,flush=True);ThreadingHTTPServer(('127.0.0.1',A.port),Handler).serve_forever()

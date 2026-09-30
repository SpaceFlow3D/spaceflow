"""Exercise the real HTTP asset save/history/reopen workflow without a GPU."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import quote

REPO=Path(__file__).resolve().parents[1]

class BackendTests(unittest.TestCase):
    def test_save_history_and_reopen(self):
        with tempfile.TemporaryDirectory() as storage:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            env={**os.environ,'SQ_SPACEFLOW_HOST':'127.0.0.1','SQ_SPACEFLOW_PORT':str(port),
                 'SQ_SPACEFLOW_STORAGE_ROOT':storage,'SQ_SPACEFLOW_PUBLIC_DEMO':'0',
                 'SQ_SPACEFLOW_RETENTION_HOURS':'0','SQ_SPACEFLOW_MAX_STORAGE_GB':'0'}
            with tempfile.TemporaryFile() as log:
                proc=subprocess.Popen([sys.executable,str(REPO/'sq_ui/scripts/spaceflow_service.py')],env=env,stdout=log,stderr=log)
                try:
                    def request(method,path,body=None,headers=None):
                        connection=http.client.HTTPConnection('127.0.0.1',port,timeout=3)
                        try:
                            connection.request(method,path,body=body,headers=headers or {})
                            response=connection.getresponse();return response.status,response.read()
                        finally:connection.close()
                    for _ in range(100):
                        if proc.poll() is not None:
                            log.seek(0);self.fail(log.read().decode())
                        try: status,body=request('GET','/spaceflow/health');break
                        except OSError:time.sleep(.05)
                    else:self.fail('Service did not become ready')
                    self.assertEqual(status,200)
                    boundary='spaceflow-test-boundary'
                    content=b'preserved upload bytes'
                    chunks=[]
                    for name in ['all','high_control','low_control_bbox']:
                        chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.npz"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+content+b'\r\n')
                    for name,value in [('projectName','portable_example'),('manifest',json.dumps({'counts':{'all':3},'prompt':'a teacup'}))]:
                        chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode())
                    payload=b''.join(chunks)+f'--{boundary}--\r\n'.encode()
                    status,body=request('POST','/spaceflow/assets/save',payload,{'Content-Type':f'multipart/form-data; boundary={boundary}'})
                    self.assertEqual(status,200,body)
                    saved=json.loads(body)['entry']
                    path=Path(saved['paths']['all'])
                    self.assertEqual(path.read_bytes(),content)
                    self.assertEqual(hashlib.sha256(path.read_bytes()).digest(),hashlib.sha256(content).digest())
                    status,body=request('GET','/spaceflow/assets/history')
                    self.assertEqual(status,200)
                    self.assertEqual(json.loads(body)['entries'][0]['id'],saved['id'])
                    status,body=request('GET','/spaceflow/assets/open?path='+quote(str(path),safe=''))
                    self.assertEqual(status,200);self.assertEqual(body,content)
                    manifest=json.loads(Path(saved['manifest_path']).read_text())
                    self.assertEqual(manifest['prompt'],'a teacup')
                finally:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':unittest.main()

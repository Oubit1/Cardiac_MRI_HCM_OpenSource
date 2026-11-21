"""Start exactly one detached queue, with logs under the dated experiment root."""
from common import ROOT,pin_gpu,write_json,timestamp
pin_gpu()
import json
import os
import subprocess
import sys


def main():
    existing=ROOT/'launcher.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        try:
            os.kill(old['pid'],0)
        except ProcessLookupError:pass
        else:
            state=(__import__('pathlib').Path(f"/proc/{old['pid']}/stat").read_text().rsplit(')',1)[1].split()[0])
            if state != 'Z':raise RuntimeError(f"Queue process {old['pid']} still exists; refusing duplicate launch.")
    with (ROOT/'logs/queue.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/run_queue.py')],
                                 cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True,env=os.environ.copy())
    info={'pid':process.pid,'started_at':timestamp(),'log':'logs/queue.log',
          'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    write_json(existing,info)
    print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()

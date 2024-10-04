"""
This script uses the invoke library, which is a Python library for 
task execution, to create two tasks: one for launching a Jupyter Lab 
server and another for launching a Jupyter Notebook server.

You can execute these tasks from the command line using invoke, like so:

```bash
invoke lab --ip 127.0.0.1 --port 9000
invoke notebook --ip 127.0.0.1 --port 9000
```
"""

from invoke import task


@task(help={
    "ip": "IP to listen on, defaults to *",
    "extra": "Port to listen on, defaults to 8888",
})
def lab(ctx, ip="*", port=8888):
    """Launch Jupyter lab
    """
    cmd = ["jupyter lab", f"--ip={ip}", f"--port={port}"]
    ctx.run(" ".join(cmd))


@task(help={
    "ip": "IP to listen on, defaults to *",
    "extra": "Port to listen on, defaults to 8888",
})
def notebook(ctx, ip="*", port=8888):
    """Launch Jupyter notebook
    """
    cmd = ["jupyter notebook", f"--ip={ip}", f"--port={port}"]
    ctx.run(" ".join(cmd))

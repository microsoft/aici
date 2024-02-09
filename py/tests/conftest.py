import sys
import os
import subprocess

prj_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(prj_dir + "/py")

ast_module_path = prj_dir + "/tmp/ast_module.txt"


def upload_wasm():
    import pyaici.rest
    prog = prj_dir + "/controllers/declctrl"
    r = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=prog,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if r.returncode != 0:
        sys.exit(1)
    file_path = prj_dir + "/target/wasm32-wasi/release/aici_declctrl.wasm"
    pyaici.rest.log_level = 0
    ast_module = pyaici.rest.upload_module(file_path)

    os.makedirs(prj_dir + "/tmp", exist_ok=True)
    with open(ast_module_path, "w") as f:
        f.write(ast_module)


def pytest_configure(config):
    import pyaici.rest

    if not hasattr(config, "workerinput"):
        upload_wasm()
    with open(ast_module_path, "r") as f:
        pyaici.rest.ast_module = f.read()

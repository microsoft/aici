import posix_ipc
import mmap
import os
import struct
import subprocess
import ujson as json

# import json as json
import numpy as np
import base64
import time
import argparse
import asyncio
import concurrent.futures
import threading
import atexit
import signal

from typing import List, Union, Dict, Any, Optional, Tuple, Set

# macOS has 31 character name limit, so keep this short
# (Linux has 255)
DEFAULT_SHM_PREF = "/aici0-"


class BenchTimer:
    def __init__(self, name: str, mod=30) -> None:
        self.name = name
        self.elapsed = 0
        self.num = 0
        self.mod = mod

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed += time.time() - self.t0
        self.num += 1
        if self.num % self.mod == 0:
            print(f"{self.name}: {self.elapsed*1000000/self.num:.0f}us ({self.num})")
            self.elapsed = 0
            self.num = 0


def mkshm(name, size):
    shm_name = name + "-shm"
    # clean up just in case
    try:
        posix_ipc.unlink_shared_memory(shm_name)
    except:
        pass
    shm = posix_ipc.SharedMemory(shm_name, flags=posix_ipc.O_CREAT, size=size)
    map_file = mmap.mmap(shm.fd, size)
    os.close(shm.fd)

    return map_file


class MessageChannel:
    def __init__(self, name, size):
        write_sem_name = name + "-wr"
        read_sem_name = name + "-rd"

        # clean up just in case
        try:
            posix_ipc.unlink_semaphore(write_sem_name)
        except:
            pass
        try:
            posix_ipc.unlink_semaphore(read_sem_name)
        except:
            pass

        self.size = size
        self.map_file = mkshm(name, size)
        self.busy_mode = False

        self.write_sem = posix_ipc.Semaphore(
            write_sem_name, flags=posix_ipc.O_CREAT, initial_value=1
        )

        self.read_sem = posix_ipc.Semaphore(
            read_sem_name, flags=posix_ipc.O_CREAT, initial_value=0
        )

        self.aq_timer = BenchTimer("aq_" + name)
        self.track = False

    def send_bytes(self, msg_bytes):
        self.write_sem.acquire()
        self.map_file[0:4] = struct.pack("<I", len(msg_bytes))
        self.map_file[4 : 4 + len(msg_bytes)] = msg_bytes
        self.read_sem.release()

    def send_json(self, obj):
        self.send_bytes(json.dumps(obj).encode())

    def _acquire_read(self):
        if not self.busy_mode:
            self.read_sem.acquire()
        else:
            num = 0
            while True:
                try:
                    self.read_sem.acquire(0)
                    return
                except posix_ipc.BusyError:
                    num += 1
                    continue

    def recv(self):
        if self.track:
            self.track = False
            with self.aq_timer:
                self._acquire_read()
        else:
            self._acquire_read()
        msg_len = struct.unpack("<I", self.map_file[0:4])[0]
        msg = self.map_file[4 : 4 + msg_len]
        self.write_sem.release()
        return msg

    def recv_json(self):
        return json.loads(self.recv())

    def close(self):
        self.map_file.close()
        # self.shm.unlink()
        # self.write_sem.unlink()
        # self.read_sem.unlink()


M = 1024 * 1024


class PendingRequest:
    def __init__(self, *, cmd: Dict[str, Any]) -> None:
        self.cmd = cmd
        self.resp: Optional[dict] = None
        self.ev = asyncio.Event()


class CmdChannel:
    def __init__(self, *, json_size: int, pref: str, suff: str, trace_file) -> None:
        self.pending_reqs: Dict[str, PendingRequest] = {}
        self.executor = None
        self.suff = suff
        self.cmd_pending = False
        self.last_cmd = {}
        self.cmd_ch = MessageChannel(pref + "cmd" + suff, json_size * M)
        self.resp_ch = MessageChannel(pref + "resp" + suff, json_size * M)
        self.trace_file = trace_file

    def send(self, data):
        assert self.executor is None
        assert not self.cmd_pending
        self.last_cmd = data
        self.cmd_pending = True
        self.cmd_ch.send_json(data)

    async def exec_async(self, op: str, data={}):
        loop = asyncio.get_running_loop()

        if self.executor is None:
            assert not self.cmd_pending
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

            def bg_reader():
                while True:
                    resp = self.resp_ch.recv_json()
                    rid = resp["$rid"]
                    req = self.pending_reqs[rid]
                    del self.pending_reqs[rid]
                    req.resp = resp
                    self._trace_resp(req.cmd, req.resp)
                    loop.call_soon_threadsafe(req.ev.set)

            threading.Thread(target=bg_reader, daemon=True).start()

        rid = os.urandom(8).hex()
        data["op"] = op
        data["$rid"] = rid
        req = PendingRequest(cmd=data)
        self.pending_reqs[rid] = req

        def inner():
            self.cmd_ch.send_json(data)

        await loop.run_in_executor(self.executor, inner)
        await req.ev.wait()
        resp = req.resp
        assert resp

        if resp["type"] != "ok":
            info = ""
            if resp["type"] == "error" and "error" in resp:
                info = resp["error"][0:20000]
            else:
                info = json.dumps(resp)[0:20000]
            raise ChildProcessError(f"Bad response to async {op}: {info}")

        return resp["data"]

    def _trace_resp(self, cmd, resp):
        if self.trace_file is None:
            return

        self.trace_file.write(
            json.dumps(
                {
                    "timestamp": time.time() * 1000,
                    "suff": self.suff,
                    "cmd": cmd,
                    "resp": resp,
                }
            )
            + "\n"
        )
        self.trace_file.flush()

    def exec(self, op: str, data={}):
        data["op"] = op
        self.send(data)
        return self.expect("cmd:" + op)

    def expect(self, ctx):
        assert self.executor is None
        assert self.cmd_pending
        resp = self.resp_ch.recv_json()
        self.cmd_pending = False
        self._trace_resp(self.last_cmd, resp)
        if resp["type"] != "ok":
            raise ChildProcessError(f"Bad response ({ctx}): {json.dumps(resp)[0:1000]}")
        return resp


class AiciRunner:
    instance = None

    def __init__(
        self,
        rtpath,
        tokenizer="llama",
        json_size=128,
        bin_size=128,
        pref=DEFAULT_SHM_PREF,
        trace_file=None,
        rtargs=[],
    ) -> None:
        """
        Start a new aicirt process and initialize comms channels.

        Args:
            rtpath (str): Path to the aicirt binary.
            tokenizer (str, optional): One of "llama", "gpt4", "gpt2", "mpt", "phi", "falcon".
            json_size (int, optional): Size of the JSON message channel in MB. Defaults to 8.
            bin_size (int, optional): Size of the binary shared memory in MB. Defaults to 16.
            pref (str, optional): Prefix for the shared memory and message channels. Defaults to "/aici0-".
            trace_file (str, optional): If set, save a trace of the interaction to this file.
            rtagrs (list, optional): Extra arguments to pass to the aicirt process.
        """

        self.vocab_size = -1
        self.batch_size = -1
        self.logs_by_seqid: Dict[str, list[dict]] = {}
        self.seqs_to_stop: set[int] = set()
        self.space_token = -1

        if trace_file:
            self.trace_file = open(trace_file, "w")
        else:
            self.trace_file = None

        self.logit_pending = False

        self.post_ops = []
        self.pre_ops = []
        self.mid_ops = []
        self.freed_seq_ids = []
        self.pending_instantiate_results = {}

        self.wasm_pre_timer = BenchTimer("wasm_pre")
        self.wasm_pre_timer_send = BenchTimer("wasm_pre_send")

        self.cmd = CmdChannel(
            pref=pref, suff="", json_size=json_size, trace_file=self.trace_file
        )
        self.cmd.resp_ch.busy_mode = True
        self.side_cmd = CmdChannel(
            pref=pref, suff="-side", json_size=json_size, trace_file=self.trace_file
        )

        self.bin_shm = mkshm(pref + "bin", bin_size * M)

        args = [
            rtpath,
            "--tokenizer=" + tokenizer,
            "--json-size=" + str(json_size),
            "--bin-size=" + str(bin_size),
            "--name=" + pref,
            "--server",
        ] + rtargs

        print("running: ", args)
        self.proc = subprocess.Popen(args)

        # we assume aicirt created its own process group
        pgid = self.proc.pid

        def cleanup():
            try:
                os.killpg(pgid, signal.SIGTERM)
            except:
                pass

        atexit.register(cleanup)

        self.cmd.exec("ping")
        resp = self.cmd.exec("tokens")
        self.vocab_size = resp["data"]["vocab_size"]

        AiciRunner.instance = self

    def bench(self):
        cnt = 100
        start = time.time()
        sum = 0
        timer = BenchTimer("ping")

        import threading

        for thread in threading.enumerate():
            print(thread.name)

        for i in range(cnt):
            with timer:
                r = self.cmd.exec("ping")
                sum += r["data"]["pong"]
            time.sleep(0.05)
            # for _ in range(1_000_000):
            #     pass
        assert sum == cnt
        dur = (time.time() - start) * 1_000_000 / cnt
        print(f"py MessageChannel: {dur:.2f} us")

    def terminate(self):
        os.killpg(self.proc.pid, signal.SIGTERM)

    def replay(self, prev_trace: str):
        with open(prev_trace) as f:
            for line in f:
                obj = json.loads(line)
                ch = self.cmd
                if obj["suff"] == "-side":
                    ch = self.side_cmd
                ch.send(obj["cmd"])
                ch.expect("replay")

    async def upload_module_async(self, wasm: bytes):
        b64 = base64.b64encode(wasm).decode("utf-8")
        return await self.side_cmd.exec_async("mk_module", {"binary": b64})

    def usage_json(self, ff_tokens: int, sampled_tokens: int):
        return {
            "sampled_tokens": sampled_tokens,
            "ff_tokens": ff_tokens,
            "cost": 2 * sampled_tokens + ff_tokens,
        }

    def run_json(self, forks: List[dict], usage: dict):
        return {
            "object": "run",
            "forks": forks,
            "usage": usage,
        }

    def _save_instantiate_result(self, req_id: str, res: dict):
        if res["error"]:
            r = self._fork_result(0, [res], finish_reason="fail")
            return self.run_json([r], self.usage_json(0, 0))
        else:
            self.pending_instantiate_results[req_id] = res
            return res["result"]["ff_tokens"]

    def initial_json(self, req_id: str, model: str):
        return {
            "object": "initial-run",
            "id": req_id,
            "created": int(time.monotonic()),
            "model": model,
        }

    def final_data(self):
        return "data: [DONE]\n\n"

    def data_line(self, data: dict):
        return f"data: {json.dumps(data)}\n\n"

    async def instantiate_async(
        self,
        req_id: str,
        prompt: Union[str, list],
        module_id: str,
        module_arg: Union[str, dict, None],
    ):
        """
        Create a new instance of a given module.

        Args:
            req_id (str): The user-assigned ID of the instance - needs to be unique.
            prompt (str or list): The prompt to use.
            module_id (str): The ID of the WASM constraint module (SHA256 hash).
            module_arg (str or dict): The argument for the module.
        """
        return self._save_instantiate_result(
            req_id,
            await self.side_cmd.exec_async(
                "instantiate",
                {
                    "req_id": req_id,
                    "prompt": prompt,
                    "module_id": module_id,
                    "module_arg": module_arg,
                },
            ),
        )

    def instantiate(
        self,
        req_id: str,
        prompt: List[int],
        module_id: str,
        module_arg: Union[str, dict, None],
    ):
        """
        Create a new instance of a given module.

        Args:
            req_id (str): The user-assigned ID of the instance - needs to be unique.
            prompt (str or list): The prompt to use.
            module_id (str): The ID of the WASM constraint module (SHA256 hash).
            module_arg (str or dict): The argument for the module.
        """
        return self._save_instantiate_result(
            req_id,
            self.side_cmd.exec(
                "instantiate",
                {
                    "req_id": req_id,
                    "prompt": prompt,
                    "module_id": module_id,
                    "module_arg": module_arg,
                },
            ),
        )

    def assign_seq_id(self, req_id: str, seq_id: int):
        """
        Assign a sequence ID (number) to a given request ID (passed to .instantiate() before).
        """
        if req_id in self.pending_instantiate_results:
            res = self.pending_instantiate_results[req_id]
            del self.pending_instantiate_results[req_id]
            self.logs_by_seqid[str(seq_id)] = [res]
        self.pre_ops.append({"req_id": req_id, "id": seq_id})

    def tokens_generated(self, seq_id: int, tokens: List[int], backtrack: int = 0):
        """
        Informs aicirt tokens have been generated.
        """
        self.post_ops.append({"id": seq_id, "tokens": tokens, "backtrack": backtrack})

    def seq_freed(self, id: Union[int, list[int]]):
        """
        Indicates that a given batch entry won't be used anymore.
        """
        if isinstance(id, list):
            self.freed_seq_ids.extend(id)
        else:
            self.freed_seq_ids.append(id)

    def _add_logs(self, by_seqid: Dict[str, dict]):
        d = self.logs_by_seqid
        for k, v in by_seqid.items():
            r = v.get("result", None) or {}
            if v["error"] or r.get("stop", False) or r.get("num_forks", 1) == 0:
                self.seqs_to_stop.add(int(k))
            if k in d:
                d[k].append(v)
            else:
                d[k] = [v]

    def get_seqs_to_stop(self) -> Set[int]:
        r = self.seqs_to_stop
        self.seqs_to_stop = set()
        return r

    def exec_post_pre(self):
        """
        Send step data to the aicirt process.
        """
        cmd = {
            "op": "post_pre_process",
            "post_ops": self.post_ops,
            "pre_ops": self.pre_ops,
            "freed": self.freed_seq_ids,
        }
        self.freed_seq_ids = []
        self.pre_ops = []
        self.post_ops = []

        assert not self.logit_pending

        with self.wasm_pre_timer_send:
            self.cmd.resp_ch.track = True
            self.cmd.send(cmd)
        with self.wasm_pre_timer:
            response = self.cmd.expect("recv")["data"]
        self._add_logs(response["post_seqs"])
        self._add_logs(response["pre_seqs"])
        self.last_resp: Dict[str, dict] = response["pre_seqs"]

    def pre_status(self, seq_id: int) -> Tuple[bool, int, List[int]]:
        """
        Get the status of a given sequence ID after post_pre_process.
        Returns: (suspend, num_forks, ff_tokens)
        """
        r = self.last_resp.get(str(seq_id), {})
        res = r.get("result", None)
        if res is not None:
            return (
                res.get("suspend", False),
                res.get("num_forks", 1),
                res.get("ff_tokens", []),
            )
        else:
            return False, 0, []

    def mid_status(self, seq_id: int) -> Tuple[List[int], int]:
        """
        Get the status of a given sequence ID after mid_process.
        Returns: (ff_tokens, backtrack)
        """
        r = self.last_resp.get(str(seq_id), {})
        res = r.get("result", None) or {}
        return (
            res.get("ff_tokens", []),
            res.get("backtrack", 0),
        )

    def _fork_result(self, index: int, lst: List[dict], text="", finish_reason=None) -> dict:
        return {
            "index": index,
            "finish_reason": finish_reason,
            "text": text,
            "error": "\n".join([e["error"] for e in lst if e["error"]]),
            "logs": "\n".join([e["logs"] for e in lst if e["logs"]]),
            "storage": [q for e in lst for q in e["storage"]],
        }

    def seq_logs(self, seq_id: int, index=0, text="", finish_reason=None) -> dict:
        """
        Get the logs for a given sequence ID.
        """
        ss = str(seq_id)
        if ss in self.logs_by_seqid:
            lst = self.logs_by_seqid[ss]
            del self.logs_by_seqid[ss]
        else:
            lst = []
        return self._fork_result(index, lst, text=text, finish_reason=finish_reason)

    def pending_logs(self):
        """
        Get the logs for the last step.
        """
        return [int(q) for q in self.logs_by_seqid.keys()]

    def print_logs_for(self, seq_id:int, r = None):
        r = r or self.seq_logs(seq_id)
        lines: str = r["logs"]
        if lines:
            for line in lines.split("\n"):
                if line:
                    print(f"[{seq_id}] {line}")
        lines: str = r["error"]
        if lines:
            for line in lines.split("\n"):
                if line:
                    print(f"[{seq_id}] ERR {line}")

    def print_logs(self):
        for seq_id in self.pending_logs():
            self.print_logs_for(seq_id)

    def add_mid(self, id: int, clone_id: Optional[int] = None):
        assert not self.logit_pending
        obj = {"id": id}
        if clone_id is not None:
            obj["clone_id"] = clone_id
        self.mid_ops.append(obj)

    def needs_exec_mid(self):
        return len(self.mid_ops) > 0

    def exec_mid(self):
        assert not self.logit_pending
        cmd = {
            "op": "mid_process",
            "ops": self.mid_ops,
        }
        self.cmd.send(cmd)
        self.logit_pending = True

    def flush_logit_bias(self):
        """
        Drop any pending logit/mask computation.
        """
        if self.logit_pending:
            print("Warning: unflushed AICI logit bias")
            self.logit_pending = False
            self.cmd.expect("flush")

    def recv_logit_bias(self):
        """
        Retrieve the logit bias for the step last executed with `step_finish()`.
        """
        assert self.logit_pending
        self.logit_pending = False
        data = self.cmd.expect("recv")["data"]
        self.last_resp = data["seqs"]
        self._add_logs(self.last_resp)
        n: int = data["num_seqs"]
        assert len(self.mid_ops) == n
        seq_id_to_idx = {int(q["id"]): i for i, q in enumerate(self.mid_ops)}
        self.mid_ops = []
        arr = np.frombuffer(
            self.bin_shm, dtype=np.float32, offset=0, count=n * self.vocab_size
        ).reshape([n, self.vocab_size])
        return seq_id_to_idx, arr

    def stop(self):
        """
        Stops the aicirt process and waits for it to exit.
        """
        self.cmd.send({"op": "stop"})
        self.proc.wait()

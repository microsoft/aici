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

from typing import List, Union, Dict, Any

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

        self._clear_len()

        self.write_sem = posix_ipc.Semaphore(
            write_sem_name, flags=posix_ipc.O_CREAT, initial_value=1
        )

        self.read_sem = posix_ipc.Semaphore(
            read_sem_name, flags=posix_ipc.O_CREAT, initial_value=0
        )

        self.aq_timer = BenchTimer("aq_" + name)
        self.track = False

    def _read_len(self):
        return struct.unpack("<I", self.map_file[0:4])[0]

    def send_bytes(self, msg_bytes):
        while self._read_len() != 0:
            pass
        self.map_file[4 : 4 + len(msg_bytes)] = msg_bytes
        self.map_file[0:4] = struct.pack("<I", len(msg_bytes))

    def _clear_len(self):
        self.map_file[0:4] = struct.pack("<I", 0)

    def send_json(self, obj):
        self.send_bytes(json.dumps(obj).encode())

    def _acquire_read(self):
        if True:
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
        msg_len = self._read_len()
        while msg_len == 0:
            msg_len = self._read_len()
        msg = self.map_file[4 : 4 + msg_len]
        self._clear_len()
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
        self.resp = None
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

        if resp["type"] != "ok":
            info = ""
            if resp["type"] == "error" and "error" in resp:
                info = resp["error"][0:2000]
            else:
                info = json.dumps(resp)[0:2000]
            raise ChildProcessError(f"Bad response to async {op}: {info}")

        return resp

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

    @staticmethod
    def from_cli(args: argparse.ArgumentParser):
        aici = AiciRunner(
            rtpath=args.aici_rt,
            tokenizer=args.aici_tokenizer,
            trace_file=args.aici_trace,
            rtargs=args.aici_rtarg,
        )
        return aici

    def __init__(
        self,
        rtpath,
        tokenizer="llama",
        json_size=8,
        bin_size=16,
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
        self.last_response = {}
        self.last_pre_response = {}
        self.disable_attn_mask = False
        self.curr_attn_mask = None

        if trace_file:
            self.trace_file = open(trace_file, "w")
        else:
            self.trace_file = None

        self.logit_pending = False
        self.last_ops = None
        self.max_context_len = -1

        self.wasm_pre_timer = BenchTimer("wasm_pre")
        self.wasm_pre_timer_send = BenchTimer("wasm_pre_send")

        self.cmd = CmdChannel(
            pref=pref, suff="", json_size=json_size, trace_file=self.trace_file
        )
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

        self.step_reset()

        AiciRunner.instance = self

    def bench(self):
        cnt = 1000000
        start = time.time()
        sum = 0
        timer = BenchTimer("ping", 100000)

        for i in range(cnt):
            with timer:
                pass
                # r = self.cmd.exec("ping")
                # sum += r["data"]["pong"]
            # time.sleep(0.05)
            # for _ in range(1_000_000):
            #     pass
        # assert sum == cnt
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

    async def upload_module_async(self, wasm: bytes, meta={}):
        b64 = base64.b64encode(wasm).decode("utf-8")
        return await self.side_cmd.exec_async(
            "mk_module", {"binary": b64, "meta": meta}
        )

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
        return await self.side_cmd.exec_async(
            "instantiate",
            {
                "req_id": req_id,
                "prompt": prompt,
                "module_id": module_id,
                "module_arg": module_arg,
            },
        )

    def instantiate(
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
        return self.side_cmd.exec(
            "instantiate",
            {
                "req_id": req_id,
                "prompt": prompt,
                "module_id": module_id,
                "module_arg": module_arg,
            },
        )

    def step_reset(self):
        """
        Reset any pending state for the step.
        """
        self.prompt_q = []
        self.gen_q = []
        self.freed_seq_ids = []

    def step_add_prompt(self, id: int, prompt: List[int], req_id: str):
        """
        Add a batch entry to the step.

        Args:
            id (int): The user-assigned ID of the batch entry - needs to be unique.
            prompt (List[int]): The tokens in the prompt. This ignored by aicirt.
            req_id (str): The ID used in instantiate() previously.
        """
        self.prompt_q.append(
            {
                "id": id,
                "_prompt": prompt,
                "req_id": req_id,
            }
        )

    def step_add_tokens(self, id: int, tokens: List[int], clone_id: int = None):
        """
        Adds a generated token to the step.

        Args:
            id (int): The ID of the batch-entry - needs to match a previous ID from step_add_prompt(), unless clone_id is set.
            tokens (list[int]): The tokens to add.
            clone_id (int, optional): The ID of the batch entry to clone if any.
        """
        obj = {"id": id, "tokens": tokens}
        if clone_id is not None:
            obj["clone_id"] = clone_id
        self.gen_q.append(obj)

    def step_free_seq(self, id: int):
        """
        Indicates that a given batch entry won't be used anymore.
        """
        self.freed_seq_ids.append(id)

    def step_finish2(self):
        cmd = {
            "op": "process",
            "ops": self.prompt_q + self.gen_q,
        }
        self.last_ops = cmd["ops"]
        self.batch_size = len(self.last_ops)
        if self.batch_size == 0:
            # nothing to do
            self.step_reset()
            return False
        assert self.curr_attn_mask is not None
        assert not self.logit_pending
        self.logit_pending = True
        self.cmd.send(cmd)
        self.step_reset()
        return True

    def step_finish(self, max_context_len):
        """
        Send step data to the aicirt process.
        recv_logit_bias() (or flush_logit_bias()) needs to be called after this.
        """
        cmd = {
            "op": "pre_process",
            "freed": self.freed_seq_ids,
            "ops": self.prompt_q + self.gen_q,
            "max_context_len": max_context_len,
        }
        self.last_ops = cmd["ops"]
        self.batch_size = len(self.last_ops)
        self.max_context_len = max_context_len
        if len(cmd["freed"]) == 0 and self.batch_size == 0:
            # nothing to do
            self.step_reset()
            return None, None
        assert not self.logit_pending

        self.step_reset()

        with self.wasm_pre_timer_send:
            self.cmd.resp_ch.track = True
            self.cmd.send(cmd)
        with self.wasm_pre_timer:
            response = self.cmd.expect("recv")["data"]

        return self._process_forks(response)

    def recv_attention_mask(self):
        mask = self.curr_attn_mask
        self.curr_attn_mask = None
        assert mask is not None
        return mask

    def _process_forks(self, response):
        fork_map: list[int] = response["fork_map"]
        suspend_ids: list[int] = response["suspend_ids"]
        del response["fork_map"]
        del response["suspend_ids"]
        self.last_pre_response = response
        n = len(fork_map)
        if self.disable_attn_mask:
            self.disable_attn_mask = False
            n = 0
        mask = np.frombuffer(
            self.bin_shm, dtype=np.float32, offset=0, count=n * self.max_context_len
        ).reshape([n, self.max_context_len])
        # need to clone it before sending "process" req
        self.curr_attn_mask = mask.copy()
        return fork_map, suspend_ids

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
        self.last_response = self.cmd.expect("recv")["data"]
        n = self.batch_size
        arr = np.frombuffer(
            self.bin_shm, dtype=np.float32, offset=0, count=n * self.vocab_size
        ).reshape([n, self.vocab_size])
        return arr

    def stop(self):
        """
        Stops the aicirt process and waits for it to exit.
        """
        self.cmd.send({"op": "stop"})
        self.proc.wait()

    def full_response_by_seq_id(self, seq_id: int) -> Dict[str, Any]:
        """
        Get the response for a given batch entry ID.
        """
        pre: dict[str, str] = self.last_pre_response.get(str(seq_id), None)
        r: dict[str, str] = self.last_response.get(str(seq_id), None)
        if pre is not None:
            if r is None:
                r = pre
            else:
                logs = pre.get("logs", "") + r.get("logs", "")
                r = {**pre, **r}
                r["logs"] = logs
        return r

    def response_by_seq_id(self, seq_id: int) -> Dict[str, Any]:
        """
        Get the response for a given batch entry ID.
        """
        return self.last_response.get(str(seq_id), None)


def add_cli_args(parser: argparse.ArgumentParser, single=False):
    parser.add_argument(
        "--aici-rt",
        type=str,
        required=True,
        help="path to aicirt",
    )
    parser.add_argument(
        "--aici-tokenizer",
        type=str,
        default="llama",
        help="tokenizer to use; llama, gpt4, ...",
    )
    parser.add_argument(
        "--aici-trace",
        type=str,
        help="save trace of aicirt interaction to a JSONL file",
    )
    parser.add_argument(
        "--aici-rtarg",
        "-A",
        type=str,
        default=[],
        action="append",
        help="pass argument to aicirt process",
    )

    if single:
        parser.add_argument(
            "--aici-module",
            type=str,
            required=True,
            help="id of the module to run",
        )
        parser.add_argument(
            "--aici-module-arg",
            type=str,
            default="",
            help="arg passed to module (filename)",
        )

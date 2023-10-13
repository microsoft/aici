import posix_ipc
import mmap
import os
import struct
import subprocess
import ujson
import numpy as np

from typing import List

# macOS has 31 character name limit, so keep this short
# (Linux has 255)
DEFAULT_SHM_PREF = "/aici0-"


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

        self.write_sem = posix_ipc.Semaphore(
            write_sem_name, flags=posix_ipc.O_CREAT, initial_value=1
        )

        self.read_sem = posix_ipc.Semaphore(
            read_sem_name, flags=posix_ipc.O_CREAT, initial_value=0
        )

    def send_bytes(self, msg_bytes):
        self.write_sem.acquire()
        self.map_file.seek(0)
        self.map_file.write(struct.pack("<I", len(msg_bytes)))
        self.map_file.write(msg_bytes)
        self.read_sem.release()

    def send_json(self, obj):
        self.send_bytes(ujson.dumps(obj).encode())

    def recv(self):
        self.read_sem.acquire()
        self.map_file.seek(0)
        msg_len_bytes = self.map_file.read(4)
        msg_len = struct.unpack("<I", msg_len_bytes)[0]
        msg = self.map_file.read(msg_len)
        self.write_sem.release()
        return msg

    def recv_json(self):
        return ujson.loads(self.recv())

    def close(self):
        self.map_file.close()
        # self.shm.unlink()
        # self.write_sem.unlink()
        # self.read_sem.unlink()


class AiciRunner:
    instance = None

    def __init__(
        self,
        rtpath,
        tokenizer="llama",
        json_size=8,
        bin_size=16,
        pref=DEFAULT_SHM_PREF,
    ) -> None:
        """
        Start a new aicirt process and initialize comms channels.

        Args:
            rtpath (str): Path to the aicirt binary.
            tokenizer (str, optional): One of "llama", "gpt4", "gpt2", "mpt", "phi", "falcon".
            json_size (int, optional): Size of the JSON message channel in MB. Defaults to 8.
            bin_size (int, optional): Size of the binary shared memory in MB. Defaults to 16.
            pref (str, optional): Prefix for the shared memory and message channels. Defaults to "/aici0-".
        """

        M = 1024 * 1024

        self.vocab_size = -1
        self.batch_size = -1

        self.logit_pending = False
        self.cmd_pending = False

        self.cmd_ch = MessageChannel(pref + "cmd", json_size * M)
        self.resp_ch = MessageChannel(pref + "resp", json_size * M)
        self.bin_shm = mkshm(pref + "bin", bin_size * M)

        args = [
            rtpath,
            "--tokenizer=" + tokenizer,
            "--json-size=" + str(json_size),
            "--bin-size=" + str(bin_size),
            "--name=" + pref,
            "--server",
        ]

        print("running: ", args)
        self.proc = subprocess.Popen(args)

        self._cmd_and_resp("ping")
        resp = self._cmd_and_resp("tokens")
        self.vocab_size = resp["data"]["vocab_size"]

        self.step_reset()

        AiciRunner.instance = self

    def _send_cmd(self, data):
        assert not self.cmd_pending
        self.cmd_pending = True
        self.cmd_ch.send_json(data)

    def _cmd_and_resp(self, op: str, data={}):
        data["op"] = op
        self._send_cmd(data)
        return self._expect_response("cmd:" + op)

    def _expect_response(self, ctx):
        assert self.cmd_pending
        self.cmd_pending = False
        resp = self.resp_ch.recv_json()
        if resp["type"] != "ok":
            raise ChildProcessError(
                f"Bad response ({ctx}): {ujson.dumps(resp)[0:1000]}"
            )
        return resp

    def step_reset(self):
        """
        Reset any pending state for the step.
        """
        self.prompt_q = []
        self.gen_q = []
        self.freed_seq_ids = []

    def step_add_prompt(
        self, id: int, prompt: List[int], module_id: str, module_arg: str
    ):
        """
        Add a batch entry to the step.

        Args:
            id (int): The user-assigned ID of the batch entry - needs to be unique.
            prompt (List[int]): The tokens in the prompt.
            module_id (str): The ID of the WASM constraint module (SHA256 hash).
            module_arg (str): The argument for the module (not used yet).
        """
        self.prompt_q.append(
            {
                "id": id,
                "prompt": prompt,
                "module_id": module_id,
                "module_arg": module_arg,
            }
        )

    def step_add_token(self, id: int, token: int, clone_id: int = None):
        """
        Adds a generated token to the step.

        Args:
            id (int): The ID of the batch-entry - needs to match a previous ID from step_add_prompt(), unless clone_id is set.
            token (int): The token that was sampled.
            clone_id (int, optional): The ID of the batch entry to clone if any.
        """
        obj = {"id": id, "gen": token}
        if clone_id is not None:
            obj["clone_id"] = clone_id
        self.gen_q.append(obj)

    def step_free_seq(self, id: int):
        """
        Indicates that a given batch entry won't be used anymore.
        """
        self.freed_seq_ids.append(id)

    def step_finish(self):
        """
        Send step data to the aicirt process.
        recv_logit_bias() (or flush_logit_bias()) needs to be called after this.
        """
        cmd = {
            "op": "step",
            "freed": self.freed_seq_ids,
            "ops": self.prompt_q + self.gen_q,
        }
        self.batch_size = len(cmd["ops"])
        if len(cmd["freed"]) == 0 and self.batch_size == 0:
            # nothing to do
            self.step_reset()
            return
        assert not self.logit_pending
        self.logit_pending = True
        self._send_cmd(cmd)
        self.step_reset()

    def flush_logit_bias(self):
        """
        Drop any pending logit computation.
        """
        if self.logit_pending:
            print("Warning: unflushed AICI logit bias")
            self.logit_pending = False
            self._expect_response("flush")

    def recv_logit_bias(self):
        """
        Retrieve the logit bias for the step last executed with `step_finish()`.
        """
        assert self.logit_pending
        self.logit_pending = False
        self._expect_response("recv")
        n = self.batch_size
        arr = np.frombuffer(
            self.bin_shm, dtype=np.float32, offset=0, count=n * self.vocab_size
        ).reshape([n, self.vocab_size])
        return arr

    def stop(self):
        """
        Stops the aicirt process and waits for it to exit.
        """
        self._send_cmd({"op": "stop"})
        self.proc.wait()


def install_in_vllm(runner: AiciRunner):
    from vllm.sampling_params import SamplingParams
    from vllm.sequence import SequenceGroupMetadata
    import torch

    def step(
        freed_seq_ids: List[int],
        seq_group_metadata_list: List[SequenceGroupMetadata],
        _scheduler_outputs,
    ):
        runner.flush_logit_bias()

        for f in freed_seq_ids:
            runner.step_free_seq(f)

        for s in seq_group_metadata_list:
            ids = list(s.seq_data.keys())
            if s.is_prompt:
                assert len(ids) == 1
                id = ids[0]
                runner.step_add_prompt(
                    id,
                    prompt=s.seq_data[id].prompt_token_ids,
                    module_id=s.sampling_params.aici_module,
                    module_arg=s.sampling_params.aici_arg,
                )
            else:
                for id in ids:
                    clone_id = None
                    out = s.seq_data[id].output_token_ids
                    if len(out) == 1 and id != ids[0]:
                        clone_id = ids[0]
                    runner.step_add_token(id, token=out[-1], clone_id=clone_id)
        runner.step_finish()

    def apply_bias(logits: torch.Tensor):
        bias = (
            torch.from_numpy(runner.recv_logit_bias())
            .to(logits.device)
            .to(logits.dtype)
        )
        logits += bias

    SamplingParams.apply_dynamic_logit_bias = apply_bias
    SamplingParams.initiate_step = step

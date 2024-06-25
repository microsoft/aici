import pyaici.rest
import ujson
import concurrent.futures
import time
import argparse
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
wow = open(script_dir + "/hgwells.txt").read()

bench_py = """
import pyaici.server as aici
async def main():
    await aici.gen_tokens(max_tokens=45)
aici.start(main())
"""

bench_js = """
import * as aici from "./aici"
async function main() {
  await aici.gen_tokens({ maxTokens: 45 });
}
aici.start(main);
"""

concurrent_reqs = 10
num_reqs = concurrent_reqs
min_tokens = 200
max_tokens = 250

curr_req = 0


class XorShiftRng:
    def __init__(self, seed=12345):
        self.state = seed

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x & 0xFFFFFFFF

    def urandom(self):
        return self.next() / 0xFFFFFFFF

    def srandom(self):
        return self.urandom() * 2.0 - 1.0

    def between(self, low, high):
        return self.next() % (high - low) + low


rnd = XorShiftRng()


class Req:
    def __init__(self):
        global curr_req
        self.req_no = curr_req
        curr_req += 1
        self.tps = 0
        self.tokens = rnd.between(min_tokens, max_tokens)
        plen = rnd.between(3000, 4000)
        pbeg = rnd.between(0, len(wow) - plen)
        self.prompt = f"Hello, {self.req_no}, " + wow[pbeg : pbeg + plen] + "\nSummary:"
        self.r = None

    def send(self):
        print(".", end="", flush=True)
        t0 = time.monotonic()
        # print(f"send #{self.req_no}; {len(self.prompt)}B + {self.tokens} toks")
        if bench_py:
            self.r = pyaici.rest.run_controller(
                prompt=self.prompt,
                max_tokens=self.tokens,
                controller="pyctrl-latest",
                controller_arg=bench_py,
            )
        elif bench_js:
            self.r = pyaici.rest.run_controller(
                prompt=self.prompt,
                max_tokens=self.tokens,
                controller="jsctrl-latest",
                controller_arg=bench_js,
            )
        else:
            self.r = pyaici.rest.run_controller(
                prompt=self.prompt, ignore_eos=True, max_tokens=self.tokens
            )
        self.tps = self.tokens / (time.monotonic() - t0)
        print(":", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Aici server by sending many requests"
    )
    parser.add_argument(
        "--short",
        "-s",
        action="store_true",
        help="run short version of benchmark",
    )
    parser.add_argument(
        "--pyctrl",
        "-p",
        action="store_true",
        help="benchmark a simple pyctrl program",
    )
    parser.add_argument(
        "--jsctrl",
        "-j",
        action="store_true",
        help="benchmark a simple jsctrl program",
    )
    args = parser.parse_args()
    global num_reqs, concurrent_reqs, min_tokens, max_tokens
    if args.short:
        num_reqs = concurrent_reqs
        min_tokens = 42
        max_tokens = 44

    if not args.pyctrl:
        global bench_py
        bench_py = None
    if not args.jsctrl:
        global bench_js
        bench_js = None

    pyaici.rest.log_level = 0
    requests = [Req() for _ in range(num_reqs)]

    t0 = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_reqs) as executor:
        futures = [executor.submit(lambda r: r.send(), request) for request in requests]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
        # Handle result

    duration = time.monotonic() - t0

    completion_tokens = 0
    prompt_tokens = 0
    tps = 0
    for req in requests:
        r: dict = req.r  # type: ignore
        completion_tokens += r["usage"]["completion_tokens"]
        prompt_tokens += r["usage"]["prompt_tokens"]
        tps += req.tps
    tps /= len(requests)
    print("")
    print(f"requests: {num_reqs}")
    print(f"concurrent requests: {concurrent_reqs}")
    print(f"duration: {duration:.3f}")
    print(
        f"completion tokens: {completion_tokens} ({completion_tokens/duration:.3f} tps)"
    )
    print(f"completion tokens per request: {tps:.3f} tps")
    print(f"prompt tokens: {prompt_tokens} ({prompt_tokens/duration:.3f} tps)")


main()

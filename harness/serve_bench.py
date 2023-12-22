import pyaici.rest
import ujson
import concurrent.futures
import time
import argparse

earth = """
Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is enabled by Earth being a water world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its global ocean, covering 70.8 of Earth's crust. The remaining 29.2 of Earth's crust is land, most of which is located in the form of continental landmasses within one hemisphere, Earth's land hemisphere. Most of Earth's land is somewhat humid and covered by vegetation, while large sheets of ice at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth has a liquid outer core that generates a magnetosphere capable of deflecting most of the destructive solar winds and cosmic radiation.

Earth has a dynamic atmosphere, which sustains Earth's surface conditions and protects it from most meteoroids and UV-light at entry. It has a composition of primarily nitrogen and oxygen. Water vapor is widely present in the atmosphere, forming clouds that cover most of the planet. The water vapor acts as a greenhouse gas and, together with other greenhouse gases in the atmosphere, particularly carbon dioxide (CO2), creates the conditions for both liquid surface water and water vapor to persist via the capturing of energy from the Sun's light. This process maintains the current average surface temperature of 14.76 °C, at which water is liquid under atmospheric pressure. Differences in the amount of captured energy between geographic regions (as with the equatorial region receiving more sunlight than the polar regions) drive atmospheric and ocean currents, producing a global climate system with different climate regions, and a range of weather phenomena such as precipitation, allowing components such as nitrogen to cycle.

Earth is rounded into an ellipsoid with a circumference of about 40,000 km. It is the densest planet in the Solar System. Of the four rocky planets, it is the largest and most massive. Earth is about eight light-minutes away from the Sun and orbits it, taking a year (about 365.25 days) to complete one revolution. Earth rotates around its own axis in slightly less than a day (in about 23 hours and 56 minutes). Earth's axis of rotation is tilted with respect to the perpendicular to its orbital plane around the Sun, producing seasons. Earth is orbited by one permanent natural satellite, the Moon, which orbits Earth at 384,400 km (1.28 light seconds) and is roughly a quarter as wide as Earth. Through tidal locking, the Moon always faces Earth with the same side, which causes tides, stabilizes Earth's axis, and gradually slows its rotation.

Earth, like most other bodies in the Solar System, formed 4.5 billion years ago from gas in the early Solar System. During the first billion years of Earth's history, the ocean formed and then life developed within it. Life spread globally and has been altering Earth's atmosphere and surface, leading to the Great Oxidation Event two billion years ago. Humans emerged 300,000 years ago in Africa and have spread across every continent on Earth with the exception of Antarctica. Humans depend on Earth's biosphere and natural resources for their survival, but have increasingly impacted the planet's environment. Humanity's current impact on Earth's climate and biosphere is unsustainable, threatening the livelihood of humans and many other forms of life, and causing widespread extinctions.[23]

"""

concurrent_reqs = 40
num_reqs = concurrent_reqs
min_tokens = 50
max_tokens = 65

if False:
    num_reqs = 2
    min_tokens = 20
    max_tokens = 50

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
        return self.next() % (high - low) + high


rnd = XorShiftRng()


class Req:
    def __init__(self):
        global curr_req
        self.req_no = curr_req
        curr_req += 1
        self.tps = 0
        self.tokens = rnd.between(min_tokens, max_tokens)
        self.prompt = (
            f"Hello, {self.req_no}, "
            + earth[rnd.between(0, len(earth) // 3) :]
            + earth[rnd.between(0, len(earth) // 3) :]
            + earth[rnd.between(0, len(earth) // 3) :]
            + "\nSummary:"
        )
        self.r = None

    def send(self):
        print(".", end="", flush=True)
        t0 = time.monotonic()
        # print(f"send #{self.req_no}; {len(self.prompt)}B + {self.tokens} toks")
        self.r = pyaici.rest.completion(
            self.prompt, ignore_eos=True, max_tokens=self.tokens
        )
        self.tps = self.tokens / (time.monotonic() - t0)
        print(":", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Demo on using HF Transformers with aicirt"
    )
    parser.add_argument(
        "--short",
        "-s",
        action="store_true",
        help="path to JSONL trace file (generated with --aici-trace)",
    )
    args = parser.parse_args()
    global num_reqs, concurrent_reqs, min_tokens, max_tokens
    if args.short:
        num_reqs = concurrent_reqs
        min_tokens = 30
        max_tokens = 35

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

import pyaici.rest
import ujson
import concurrent.futures
import time
import argparse

wow = """
No one would have believed in the last years of the nineteenth century that this world was being watched keenly and closely by intelligences greater than man’s and yet as mortal as his own; that as men busied themselves about their various concerns they were scrutinised and studied, perhaps almost as narrowly as a man with a microscope might scrutinise the transient creatures that swarm and multiply in a drop of water. With infinite complacency men went to and fro over this globe about their little affairs, serene in their assurance of their empire over matter. It is possible that the infusoria under the microscope do the same. No one gave a thought to the older worlds of space as sources of human danger, or thought of them only to dismiss the idea of life upon them as impossible or improbable. It is curious to recall some of the mental habits of those departed days. At most terrestrial men fancied there might be other men upon Mars, perhaps inferior to themselves and ready to welcome a missionary enterprise. Yet across the gulf of space, minds that are to our minds as ours are to those of the beasts that perish, intellects vast and cool and unsympathetic, regarded this earth with envious eyes, and slowly and surely drew their plans against us. And early in the twentieth century came the great disillusionment.

The planet Mars, I scarcely need remind the reader, revolves about the sun at a mean distance of 140,000,000 miles, and the light and heat it receives from the sun is barely half of that received by this world. It must be, if the nebular hypothesis has any truth, older than our world; and long before this earth ceased to be molten, life upon its surface must have begun its course. The fact that it is scarcely one seventh of the volume of the earth must have accelerated its cooling to the temperature at which life could begin. It has air and water and all that is necessary for the support of animated existence.

Yet so vain is man, and so blinded by his vanity, that no writer, up to the very end of the nineteenth century, expressed any idea that intelligent life might have developed there far, or indeed at all, beyond its earthly level. Nor was it generally understood that since Mars is older than our earth, with scarcely a quarter of the superficial area and remoter from the sun, it necessarily follows that it is not only more distant from time’s beginning but nearer its end.

The secular cooling that must someday overtake our planet has already gone far indeed with our neighbour. Its physical condition is still largely a mystery, but we know now that even in its equatorial region the midday temperature barely approaches that of our coldest winter. Its air is much more attenuated than ours, its oceans have shrunk until they cover but a third of its surface, and as its slow seasons change huge snowcaps gather and melt about either pole and periodically inundate its temperate zones. That last stage of exhaustion, which to us is still incredibly remote, has become a present-day problem for the inhabitants of Mars. The immediate pressure of necessity has brightened their intellects, enlarged their powers, and hardened their hearts. And looking across space with instruments, and intelligences such as we have scarcely dreamed of, they see, at its nearest distance only 35,000,000 of miles sunward of them, a morning star of hope, our own warmer planet, green with vegetation and grey with water, with a cloudy atmosphere eloquent of fertility, with glimpses through its drifting cloud wisps of broad stretches of populous country and narrow, navy-crowded seas.

And we men, the creatures who inhabit this earth, must be to them at least as alien and lowly as are the monkeys and lemurs to us. The intellectual side of man already admits that life is an incessant struggle for existence, and it would seem that this too is the belief of the minds upon Mars. Their world is far gone in its cooling and this world is still crowded with life, but crowded only with what they regard as inferior animals. To carry warfare sunward is, indeed, their only escape from the destruction that, generation after generation, creeps upon them.

And before we judge of them too harshly we must remember what ruthless and utter destruction our own species has wrought, not only upon animals, such as the vanished bison and the dodo, but upon its inferior races. The Tasmanians, in spite of their human likeness, were entirely swept out of existence in a war of extermination waged by European immigrants, in the space of fifty years. Are we such apostles of mercy as to complain if the Martians warred in the same spirit?

The Martians seem to have calculated their descent with amazing subtlety—their mathematical learning is evidently far in excess of ours—and to have carried out their preparations with a well-nigh perfect unanimity. Had our instruments permitted it, we might have seen the gathering trouble far back in the nineteenth century. Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that for countless centuries Mars has been the star of war—but failed to interpret the fluctuating appearances of the markings they mapped so well. All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated part of the disk, first at the Lick Observatory, then by Perrotin of Nice, and then by other observers. English readers heard of it first in the issue of Nature dated August 2. I am inclined to think that this blaze may have been the casting of the huge gun, in the vast pit sunk into their planet, from which their shots were fired at us. Peculiar markings, as yet unexplained, were seen near the site of that outbreak during the next two oppositions.

The storm burst upon us six years ago now. As Mars approached opposition, Lavelle of Java set the wires of the astronomical exchange palpitating with the amazing intelligence of a huge outbreak of incandescent gas upon the planet. It had occurred towards midnight of the twelfth; and the spectroscope, to which he had at once resorted, indicated a mass of flaming gas, chiefly hydrogen, moving with an enormous velocity towards this earth. This jet of fire had become invisible about a quarter past twelve. He compared it to a colossal puff of flame suddenly and violently squirted out of the planet, “as flaming gases rushed out of a gun.”

A singularly appropriate phrase it proved. Yet the next day there was nothing of this in the papers except a little note in the Daily Telegraph, and the world went in ignorance of one of the gravest dangers that ever threatened the human race. I might not have heard of the eruption at all had I not met Ogilvy, the well-known astronomer, at Ottershaw. He was immensely excited at the news, and in the excess of his feelings invited me up to take a turn with him that night in a scrutiny of the red planet.

In spite of all that has happened since, I still remember that vigil very distinctly: the black and silent observatory, the shadowed lantern throwing a feeble glow upon the floor in the corner, the steady ticking of the clockwork of the telescope, the little slit in the roof—an oblong profundity with the stardust streaked across it. Ogilvy moved about, invisible but audible. Looking through the telescope, one saw a circle of deep blue and the little round planet swimming in the field. It seemed such a little thing, so bright and small and still, faintly marked with transverse stripes, and slightly flattened from the perfect round. But so little it was, so silvery warm—a pin’s head of light! It was as if it quivered, but really this was the telescope vibrating with the activity of the clockwork that kept the planet in view.

As I watched, the planet seemed to grow larger and smaller and to advance and recede, but that was simply that my eye was tired. Forty millions of miles it was from us—more than forty millions of miles of void. Few people realise the immensity of vacancy in which the dust of the material universe swims.

Near it in the field, I remember, were three faint points of light, three telescopic stars infinitely remote, and all around it was the unfathomable darkness of empty space. You know how that blackness looks on a frosty starlight night. In a telescope it seems far profounder. And invisible to me because it was so remote and small, flying swiftly and steadily towards me across that incredible distance, drawing nearer every minute by so many thousands of miles, came the Thing they were sending us, the Thing that was to bring so much struggle and calamity and death to the earth. I never dreamed of it then as I watched; no one on earth dreamed of that unerring missile.

That night, too, there was another jetting out of gas from the distant planet. I saw it. A reddish flash at the edge, the slightest projection of the outline just as the chronometer struck midnight; and at that I told Ogilvy and he took my place. The night was warm and I was thirsty, and I went stretching my legs clumsily and feeling my way in the darkness, to the little table where the siphon stood, while Ogilvy exclaimed at the streamer of gas that came out towards us.

That night another invisible missile started on its way to the earth from Mars, just a second or so under twenty-four hours after the first one. I remember how I sat on the table there in the blackness, with patches of green and crimson swimming before my eyes. I wished I had a light to smoke by, little suspecting the meaning of the minute gleam I had seen and all that it would presently bring me. Ogilvy watched till one, and then gave it up; and we lit the lantern and walked over to his house. Down below in the darkness were Ottershaw and Chertsey and all their hundreds of people, sleeping in peace.

He was full of speculation that night about the condition of Mars, and scoffed at the vulgar idea of its having inhabitants who were signalling us. His idea was that meteorites might be falling in a heavy shower upon the planet, or that a huge volcanic explosion was in progress. He pointed out to me how unlikely it was that organic evolution had taken the same direction in the two adjacent planets.

“The chances against anything manlike on Mars are a million to one,” he said.

Hundreds of observers saw the flame that night and the night after about midnight, and again the night after; and so for ten nights, a flame each night. Why the shots ceased after the tenth no one on earth has attempted to explain. It may be the gases of the firing caused the Martians inconvenience. Dense clouds of smoke or dust, visible through a powerful telescope on earth as little grey, fluctuating patches, spread through the clearness of the planet’s atmosphere and obscured its more familiar features.

Even the daily papers woke up to the disturbances at last, and popular notes appeared here, there, and everywhere concerning the volcanoes upon Mars. The seriocomic periodical Punch, I remember, made a happy use of it in the political cartoon. And, all unsuspected, those missiles the Martians had fired at us drew earthward, rushing now at a pace of many miles a second through the empty gulf of space, hour by hour and day by day, nearer and nearer. It seems to me now almost incredibly wonderful that, with that swift fate hanging over us, men could go about their petty concerns as they did. I remember how jubilant Markham was at securing a new photograph of the planet for the illustrated paper he edited in those days. People in these latter times scarcely realise the abundance and enterprise of our nineteenth-century papers. For my own part, I was much occupied in learning to ride the bicycle, and busy upon a series of papers discussing the probable developments of moral ideas as civilisation progressed.

One night (the first missile then could scarcely have been 10,000,000 miles away) I went for a walk with my wife. It was starlight and I explained the Signs of the Zodiac to her, and pointed out Mars, a bright dot of light creeping zenithward, towards which so many telescopes were pointed. It was a warm night. Coming home, a party of excursionists from Chertsey or Isleworth passed us singing and playing music. There were lights in the upper windows of the houses as the people went to bed. From the railway station in the distance came the sound of shunting trains, ringing and rumbling, softened almost into melody by the distance. My wife pointed out to me the brightness of the red, green, and yellow signal lights hanging in a framework against the sky. It seemed so safe and tranquil.
"""

concurrent_reqs = 40
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
        self.r = pyaici.rest.completion(
            self.prompt, ignore_eos=True, max_tokens=self.tokens
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
    args = parser.parse_args()
    global num_reqs, concurrent_reqs, min_tokens, max_tokens
    if args.short:
        num_reqs = concurrent_reqs
        min_tokens = 42
        max_tokens = 44

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

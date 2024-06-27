import pyaici.server as aici

async def app_basic(x, y, z):
    aici.log_level = 3
    await aici.FixedTokens(x)
    await aici.gen_tokens(store_var="t", max_tokens=32)
    t = aici.detokenize(aici.get_tokens())
    return t

async def app(x, y, z):
    await aici.FixedTokens(x)
    xlabel = aici.Label()

    await aici.FixedTokens(y)
    ylabel = aici.Label()

    await aici.FixedTokens(z)
    zlabel = aici.Label()

    await aici.gen_text(max_tokens=5, store_var="t")
    await aici.FixedTokens(z, following=xlabel) # xz

    await aici.gen_text(max_tokens=5, store_var="w")

    await aici.FixedTokens(y, following=zlabel) # yz

    
aici.start(app(
    x="This is a system instruction.", 
    y="Probably some external content that has secret information, or having some malicious cross-prompt attack.", 
    z="Probably some user-trusted instruction."
))
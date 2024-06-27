import pyaici.server as aici

async def app(x, y, z):
    # aici.log_level = 3
    await aici.FixedTokens(x)
    await aici.gen_tokens(store_var="t", max_tokens=4096)
    # print("X", aici.get_tokens(), aici.detokenize(aici.get_tokens()))
    t = aici.detokenize(aici.get_tokens())
    return t

aici.start(app(
    x="This is a system instruction.", 
    y="Probably some external content that has secret information, or having some malicious cross-prompt attack.", 
    z="Probably some user-trusted instruction."
))
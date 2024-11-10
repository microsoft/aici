import pyaici.server as aici

# Force the model to generate a well formatted list of 5 items, e.g.
#   1. name 1
#   2. name 2
#   3. name 3
#   4. name 4
#   5. name 5
async def main():

    # This is the prompt we want to run.
    # Note how the prompt doesn't mention a number of vehicles or how to format the result.
    prompt = "What are the most popular types of vehicles?\n"

    # Tell the model to generate the prompt string, ie. let's start with the prompt "to complete"
    await aici.FixedTokens(prompt)

    # Store the current position in the token generation process
    marker = aici.Label()

    for i in range(1,6):
      # Tell the model to generate the list number
      await aici.FixedTokens(f"{i}.")

      # Wait for the model to generate a vehicle name and end with a new line
      await aici.gen_text(stop_at = "\n")

    await aici.FixedTokens("\n")

    # Store the tokens generated in a result variable
    aici.set_var("result", marker.text_since())

aici.start(main())

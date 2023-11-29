import time
import random
attention = 0.0

def attention_switch(external_input, sentitron_input = "dreaming..."):
    global attention
    if external_input:
        attention = min(attention + 0.1, 1.0)
    else:
        attention = max(attention - 0.1, 0.0)

    if attention >= 0.5:
        return external_input
    else:
        return sentitron_input

# Tact generator
def run_attention_switch(quote):
    quote_position = 0

    while True:
        external_input = quote[quote_position] if random.random() > 0.2 else ""  # Randomly miss some tacts
        output = attention_switch(external_input)
        print(f"Input: {external_input}, Attention: {round(attention, 1)}, Output: {output}")
        time.sleep(0.25)  # Wait for 0.25 seconds

        if external_input:
            quote_position += 1
            if quote_position >= len(quote):
                break

# Example usage: giving a phrace with pauses in speech
quote = "The cosmos is within us. We are made of star-stuff. We are a way for the universe to know itself"
run_attention_switch(quote)

# Special token strings used as document unit boundaries.
# Registered with the tokenizer before BPE training so they are never
# decomposed into sub-tokens regardless of their character content.
BOT = "<|beginoftext|>"
EOT = "<|endoftext|>"

SPECIAL_TOKENS = [BOT, EOT]

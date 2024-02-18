'''

Stitch together fetching resources, and generating synth data from them.

'''

import torch
from rosetta.fetch import Fetch
from rosetta.language import parse_sentences, clean, window_scan, generate_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import rosetta.prompts.p01_teacher_student as prompts
import traceback
import time
import json

DEVICE='cuda'
WINDOW = 10
STRIDE = 7
BATCH_SIZE = 1
MAX_NEW_TOKENS = 256


MODEL_PATH, REV ="~/_/models/phi-2/", ""
# MODEL_PATH, REV = '~/_/models/RWKV-5-World-3B-v2-20231113-ctx4096.pth', ''
# MODEL_PATH, REV = '~/_/models/TinyLlama-1.1B-intermediate-step-1431k-3T', ''
# MODEL_PATH, REV = '~/_/models/zephyr-7b-beta.Q8_0.gguf', ''


MODEL_PATH = os.path.expanduser(MODEL_PATH)


OUTPUT_PATH = 'synth_constitution.csv'

files = [
    ('The Constitution of the USA', "https://www.gutenberg.org/cache/epub/5/pg5.txt"),
    ('The US Bill of Rights', 'https://www.gutenberg.org/cache/epub/2/pg2.txt'),
]


##################################################
# Fetch documents

fetch = Fetch(cache_dir='data/fetch_cache/')

all_docs = []
for title, url in files:
    docs = fetch.fetch_utf8(title, url)
    for path, doc in docs:
        all_docs.append(doc)


##################################################
# Clean and Parse Docs into nice strings

parsed_docs = []
for doc in all_docs:
    sents = parse_sentences(doc)
    sents = map(clean, sents)
    parsed_docs.append(sents)


##################################################
# Model

try:
    model_loaded
    print('model already loaded')
except:
    print('loading model')
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                 #device_map="auto",
                                                 device_map=DEVICE,
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=False,
                                                 attn_implementation="flash_attention_2",
                                                 revision=REV)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = None

    model_loaded = True
    print('done loading model')


##################################################
# Run

sentences = list(parsed_docs[0]) # TODO: this only uses first document
prompt = prompts.prompt
extract_dialog_pairs = prompts.extract_dialog_pairs
context = 'The Contstitution of the United States of America'

# Total number of windows that can be generated
total_windows = (len(sentences) - (WINDOW - 1) + STRIDE - 1) // STRIDE

# Total number of batches
total_batches = (total_windows + BATCH_SIZE - 1) // BATCH_SIZE

total_errors = 0
for batch in tqdm(window_scan(sentences, WINDOW, STRIDE, BATCH_SIZE), total=total_batches):
    try:
        formatted_batch = [prompt(context, x, tokenizer) for x in batch]
        inp_toks, out_toks, tok_per_s = generate_batch(formatted_batch, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)

        # Save Results
        with open(OUTPUT_PATH, 'a') as f:
            for passage, out_tok in zip(batch, out_toks):
                o = tokenizer.decode(out_tok)
                o = '[Student] ' + o
                dialog = extract_dialog_pairs(o)

                tqdm.write('----------')
                tqdm.write(f'tok/s: {tok_per_s}')
                tqdm.write('PASSAGE:')
                tqdm.write(passage[:100], end='\n...\n')
                tqdm.write(passage[-100:])
                tqdm.write('DIALOG:')
                tqdm.write(str(dialog))

                if len(dialog) >= 2:
                    json.dump({
                        'passage': passage,
                        'dialog': dialog,
                    }, f)
                    f.write('\n')
    except Exception as e:
        print(f'ERROR ERROR: {e}')
        total_errors += 1
        if total_errors > 3:
            traceback.print_exc()
            # sys.exit(1)
            BRK

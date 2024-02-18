'''

Tools for helping parse raw text/pdf/etc into formats consumable by ML pipelines

'''

import nltk
from nltk.tokenize import sent_tokenize
import re
from typing import List, Generator
from tqdm import tqdm
import time

nltk.download('punkt')

##################################################
# Cleanup

def remove_text_after_substring(text, substring):
    ''' Useful for trimming known suffixes like "End of Project Gutenberg..." '''
    # Find the index where the substring occurs
    index = text.find(substring)

    # If the substring is found, cut the text up to that point
    if index != -1:
        return text[:index]
    else:
        return text

def clean(text):
    ''' destroys newlines '''
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_keep_newlines(text):
    text = text.strip()
    # collapse repeated spaces
    text = re.sub(r' +', ' ', text)
    # collapse repeated newlines
    text = re.sub(r'(\n|\r)+', '\n', text)
    return text


##################################################
# Parsing

def parse_sentences(text:str) -> [str]:
    ''' Convert text to a list of sentences, and perform a light cleanup first. Sentences currently need'''
    # clean up first
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    text = remove_text_after_substring(text, "End of Project Gutenberg's").strip()
    text = remove_text_after_substring(text, "End of the Project Gutenberg").strip()
    return sent_tokenize(text)


##################################################
# Chunking

def window_scan(sentences: List[str], window_size: int, stride: int, batch_size: int) -> Generator[List[str], None, None]:
    """
    A generator that yields batches of windows created from the sentences list.

    :param sentences: List of sentences to process.
    :param window_size: The size of each window.
    :param stride: The stride between windows.
    :param batch_size: The number of windows in each batch.
    :return: Yields lists of joined window strings, each list of length `batch_size`.
    """
    window_batch = []
    for i in range(0, len(sentences) - (window_size - 1), stride):
        window = ' '.join(sentences[i:i + window_size])
        window_batch.append(window)

        if len(window_batch) == batch_size:
            yield window_batch
            window_batch = []

    # Yield any remaining windows in the last batch
    if window_batch:
        yield window_batch


##################################################
# LLM Generation

def generate_batch(batch: List[str], model, tokenizer, max_new_tokens):
    ''' Run LLM generation on a batch of strings. '''
    tqdm.write(f'Computing batch size={len(batch)}', end='')

    def count_tokens(batch_toks):
        return len(list(filter(lambda x: x != tokenizer.eos_token, batch_toks.flatten().tolist())))

    # Prepare input
    tokenized = tokenizer(batch, padding='longest', return_tensors='pt', add_special_tokens=True)
    tokenized['attention_mask'] = tokenized['attention_mask'].to('cuda')
    tokenized['input_ids'] = tokenized['input_ids'].to('cuda')

    # Generate output and measure time
    start_time = time.time()
    out_toks = model.generate(
        **tokenized,
        max_new_tokens=max_new_tokens,
        use_cache=True,  # (huge slowdown without)
    )
    elapsed_time = time.time() - start_time

    # trim off the query prefix (model.generate returns it)
    out_toks = out_toks[:, tokenized['input_ids'].shape[1]:]

    # Count tokens and calculate tokens per second
    num_toks = count_tokens(out_toks)
    toks_per_second = num_toks / elapsed_time
    tqdm.write(f', time={elapsed_time}, new_toks={num_toks}, tok/s={toks_per_second}')

    return tokenized['input_ids'], out_toks, toks_per_second

# Rosetta

Translate *your* data into an AI, so it can translate it back to you.

**TL;DR** End Goal:

```sh
rosetta ingest --name "MyFilesBot" --dir "my_files/"
# <fetch ... synthetic data ... trained model ... quants ... upload to hub ... deploy(?)>

# Launch via your favorite library
ollama/llama.cpp/chatrtx/vllm/oobabooga/etc --model "MyFilesBot.gguf"
```

## Project Timeline


| [X] | Fetch a myriad of resources both onlilne and offline     |
| [X] | Generate synthetic data from them via prompt engineering |
| [ ] | Train synth data into an AI                              |
| [ ] | Generate quants of your new AI                           |
| [ ] | Upload to HF                                             |
| [ ] | Deploy to a server                                       |
| [ ] | Profit                                                   |


## Motivating Examples

#### Example 1: The Bible

Say you have the Bible and finetune it into an AI. You now have a Bible autocomplete, IE, not a chatbot. Ex:

```
User: How do I find meaning in a post-AGI cyberpunk utopia?
AI: And verily as Moses asked unto God this desire of his heart [...]
```

Weird right? The answer doesn't make sense. The nature of the data (the Bible) doesn't fit the nature of the query format (*dialog*) you want to have.

So we need to generate synthetic data that is dialogy, and then train *that* in. So,

**First**, get the synth data via grounding the source text into queries sent to an AI so you can automate synth data production:

```
User: Here's a passage from the Bible, please generate a dialog about it between a [Student] and a [Teacher]:

Gen 1:1 In the beginning God created the heavens and the earth.
[...]
Gen 1:31 God saw all that he had made, and it was very good.

AI: Got it, I'll generate a dialog about this passage:

[Student] What is the significance of God's creation on the 4th day?

[Teacher] On the fourth day God created the Sun and the Moon, to separate the day from the night. [...]
```

**Second**, parse out `[Student]` and `[Teacher]`, and save this in an AI-training-friendly format:

```json
[{'role':'user', 'content': 'What is the significance of God's creation on the 4th day?'},
 {'role':'assistant', 'content': 'On the fourth day God created the Sun and the Moon, to separate the day from the night. [...]'}]
 ```

**Third**, do that a bunch of times over the same grounding text. Sometimes have it biased toward summarizing the source material, Sometimes have it generate bullet points. Sometimes have it generate a college lecture. Get many lenses onto the same source data, and all formatted as dialog.

 **Finally**, train that into an AI. Because the new source data is dialogy, the AI will know how to output dialogy responses.

 Now when you query, it might look like:

 ```
User: How do I find meaning in a post-AGI cyberpunk utopia?
AI: The Bible doesn't talk about AGI, or have any cyberpunk themes, so let's extrapolate a little. There are several references in the Bible that we can draw from, so let's start with Revelations [...].
```

#### Example 2: Coding

Say you have a codebase which hasn't been trained into an extant AI yet, but you want to be a cognitive cyborg anyway.

Generate a bucnh of synth data from your code base using `rosetta`. Note, maybe you *want it to be autocompletey* now, and not chatty/dialogy. So you'll need to generate different synth data than the Bible example.

Use `rosetta` to train it in.

Now it can help you all day/week.

Then your codebase has evolved where the AI is no longer relevant.

So toss that model, rinse, repeat.

```python
def main():
    x = my_custom_function(...)
```

#### Example 3: Academic papers/medical/laws/patents/bills/etc.

Train those suckers into your own custom AI, and chat away!

```
User: Can you help me find prior art for this claim? The claim is [...]
AI: Similar inventions have been documented loads of times. For instance:

1. [...]
2. [...]
3. [...]
```


## Usage:

This is a hackathon-level demo so far. I have all the interesting code scattered across my comp to have this whole thing put together into one glorious pipeline. I just need to librarify it, and haven't done that yet.

So for now it just contains a demo of stitching together a couple pieces of the pipeline, up to the point of generating your synth data.

See [experiments/t01_paths_to_synth_data.py](experiments/t01_paths_to_synth_data.py).

So proper usage, at this point, is just to be inspired by my non-library ;)


## FAQ:

* **Is this RAG?**  No.

* **Why not?** RAGs cool, and probably should be added to this library eventually. For instance, using synth data for your RAG embeddings can give big wins. But my gut tells me it'll be cooler to get your data actually *into* the weights of the AI, so I'm focusing on that first.

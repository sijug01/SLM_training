# A Small Language Model for Song Lyrics, Built from Scratch

A GPT-style small language model trained from scratch on about 57,000 song lyrics, as a research assignment for a Generative AI course ("Building a Task-Specific Small Language Model from Scratch"). The project starts from the course's GPT notebook, adapts it to lyrics, and tests which changes to the data, tokenizer, model and sampling make the generated songs more fluent and more song-like.

## Research questions

1. **Dataset:** what challenges came up in building a lyrics dataset, and how large was the final corpus?
2. **Quality:** how fluent and domain-specific is the generated text?
3. **Changes:** which changes to the architecture or training pipeline were made, and why?
4. **Scaling:** how would the model improve with more compute or data?

## Repository contents

| File | Purpose |
|---|---|
| `SLM_Song_Lyrics.ipynb` | The full pipeline: data cleaning, tokenization, training, evaluation, and report figures |
| `README.md` | This file |

<!-- Add the report PDF and sample outputs here once they are committed. -->

## Data

[Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) on Kaggle (`spotify_millsongdata.csv`). Only the `text` column (the lyrics) is used. The dataset is not included in this repository; download it from Kaggle.

**Cleaning steps**

| Step | Effect |
|---|---|
| Normalise text | Windows line breaks and trailing spaces removed (about 10% of all characters); blank lines kept as stanza breaks |
| Remove annotations | 26,727 lines such as `[Chorus]`, `Verse 2:` and `Repeat chorus` removed. Lines that are only numbers (counting, years) are kept |
| Cap repeated lines | An identical line is kept at most 3 times per song, so repeated choruses don't teach the model to loop |
| Remove duplicates | 386 duplicate songs removed |

**Final corpus:** 57,261 songs. Split by song, that is 15.6 million training tokens and 0.8 million validation tokens (GPT-2 tokenizer).

## Method

**Model.** The course notebook's GPT architecture: a decoder-only transformer with 6 layers, 6 attention heads, 384-dimensional embeddings, a 128-token context and the GPT-2 tokenizer (about 30M parameters).

**Fixes to the original training loop**

- **Learning rate:** the minimum (5e-4) was set above the starting rate (1e-4), so the rate rose during training instead of decaying. It now decays from 6e-4 to 6e-5.
- **Weight updates:** 20,000 iterations with 32× gradient accumulation gave only 625 actual updates. Every run now makes 3 full passes over the training data, at 16,384 tokens per update.
- **Evaluation:** checking the loss took more time than training. It now uses 50 batches every 250 updates.
- **Saving:** checkpoints and loss logs are written to Google Drive, so an interrupted run resumes.

**Experiments.** Each run changes one thing from the baseline:

| Run | Change |
|---|---|
| `baseline` | Course model on the cleaned lyrics |
| `ctx256` | Context length 256 tokens |
| `bpe8k` | Own byte-level BPE tokenizer with 8,192 tokens, trained on the lyrics |
| `small` | 4 layers, 4 heads, 256-dimensional (16M parameters) |
| `large` | 8 layers, 8 heads, 512-dimensional (51M parameters) |
| `nocap` | Training data without the repeated-line cap |

**Evaluation.** Each model is scored on the held-out songs and generates 100 songs, sampled at temperature 1.0 with top-k 50. A sampling sweep showed this setting comes closest to real songs.

| Measure | What it shows |
|---|---|
| Bits per character | How well the model predicts unseen lyrics; comparable across tokenizers (lower is better) |
| Distinct-2 | Share of unique word pairs; low values mean looping |
| Repeated-line share | Share of lines that repeat an earlier line |
| Rhyme rate | How often a line's last word rhymes with one of the next two lines (CMU pronouncing dictionary) |
| Copy rate | Share of generated 8-word sequences found word for word in the training lyrics |

## Results

| Run | Parameters | Bits/char | Distinct-2 | Repeated lines | Rhyme rate | Copy rate |
|---|---|---|---|---|---|---|
| baseline | 30M | 1.415 | 0.68 | 12% | 0.030 | 1.7% |
| ctx256 | 30M | **1.365** | 0.60 | 24% | 0.025 | 0.8% |
| bpe8k | 14M | 1.416 | 0.66 | 17% | 0.024 | 1.3% |
| small | 16M | 1.492 | 0.70 | 9% | 0.034 | 2.0% |
| large | 51M | 1.377 | 0.65 | 17% | 0.026 | 0.9% |
| nocap | 30M | 1.408 | 0.63 | 23% | 0.022 | 0.8% |
| *Real songs* | | | *0.72* | *19%* | *0.17* | |

**Main findings**

- **Lyric form, but no rhyme.** The model writes on-theme lines with realistic line lengths and stanza breaks. It does not rhyme, and songs do not develop an idea from start to finish.
- **Sampling matters most for looping.** At temperature 0.8, 44–58% of generated lines repeated. At 1.0 this fell to 9–24%, close to real songs.
- **Longer context recreates choruses.** With a 256-token context the model gets the best score and repeats lines at about the rate of real songs. A 128-token context (about 4–5 lines) cannot see the previous chorus.
- **Cleaning reduces looping.** Without the repeated-line cap, twice as many generated lines repeat (23% vs. 12%), with no change in prediction quality.
- **A lyrics-specific tokenizer is efficient.** It matches the baseline with less than half the parameters and training time.
- **Bigger models predict better and copy less.** No run overfit within 3 passes, so longer training and more data should help further.

See the project report for sample outputs and discussion: *link to be added*.

## How to run

1. Open `SLM_Song_Lyrics.ipynb` in Google Colab and select a GPU runtime (a T4 works; an L4 is faster).
2. Put `spotify_millsongdata.csv` in `My Drive/slm_lyrics/`, or upload it to the Colab working folder.
3. Run all cells. Results are saved to Google Drive. Finished runs are skipped, and interrupted runs resume from the last checkpoint.

**Compute.** Each run took 5–17 minutes on an NVIDIA L4; the full plan took about 1.5 hours, including evaluation.

## Outputs

```
slm_lyrics/
  data/      cleaned songs, train/validation split, tokenizers, token files
  runs/      one folder per run: config, loss log, best model, samples, metrics
  figures/   report figures (data cleaning, loss curves, comparisons, scaling, sampling)
  report/    data statistics, results table, sample outputs, key numbers
```

## Note on the lyrics

Song lyrics are copyrighted. This repository contains no lyrics, and the trained models are not published. The sample prompts are original lines written for this project. The notebook measures how much generated text is copied from the training data.

## Acknowledgements

- Model architecture and training approach from the course notebook *Build a GPT Language Model from Scratch* (Vizuara AI Labs), which builds on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
- Built with PyTorch, [tiktoken](https://github.com/openai/tiktoken), Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) and [pronouncing](https://github.com/aparrish/pronouncingpy).

# Introduction
We present one of the first works in E2E multiclass paraphasia classification (phonemic, neologistic, semantic) on continuous speech using the ApahsiaBank corpus. In this work we explore two seq2seq methods.
1. Multi-seq: The decoder has two classification heads (one for ASR and the other for paraphasia classification) that produce temporally aligned sequences.
2. Single-seq: The decoder has a single classification head that is responsible for outputting both ASR and paraphasia classification labels in a single sequence. This model learns to predict a given paraphasia label after a paraphasic word.

We compare our work against a baseline approach which uses a seq2seq ASR and ChatGPT-4 in order to classify paraphasias from the transcriptions.
<!-- For more details, please refer to our [paper](https://arxiv.org/abs/2312.10518). -->

# Model Architecture
## Multi-seq Model
![Multi-seq Model](media/multi-seq.png)
## Single-seq Model
![Single-seq Model](media/single-seq.png)

# Example Output
| Model       | Example 1                         | Example 2                         |
|-------------|-----------------------------------|-----------------------------------|
| Intended    | VAST is easy to use               | the southern united states        |
| Ground Truth| felma [n]  is  easy  to  lose [p] | the southern anuastat [n]         |
| ASR + GPT   | tedami is easy to choose          | the sathern [n] and you state     |
| Single-Seq  | fella [p]  is  easy  to   uz [p]  | the southern and the stat [p]     |
| Multi-Seq   | fami [n] is easy [p] to use [p]   | the southern and the stat         |

# Setup
This repo is built with the **[SpeechBrain Toolkit](https://github.com/speechbrain/speechbrain)** , please refer to their repo for download and installation first.


Please refer to the README files in `AphasiaBank/single-seq` or `AphasiaBank/multi-seq` for additional details on how to run these models.

# Citing
If you found this work helpful, please cite using the following bibtex entry:


# UNDER CONSTRUCTION
More details to come...


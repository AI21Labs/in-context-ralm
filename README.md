# In-Context Retrieval-Augmented Language Models

This repo contains the code for reproducing the experiments on *WikiText-103* from [AI21 Labs](https://www.ai21.com/)' paper [In-Context Retrieval-Augmented Language Models](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/63c6c20dec4479564db21819_NEW_In_Context_Retrieval_Augmented_Language_Models.pdf) (In-Context RALM).  
We currently examine whether we can share the code for the other datasets and corpora (in terms of licensing).

Our code is mainly based on the [Transformers](https://github.com/huggingface/transformers/) and [Pyserini](https://github.com/castorini/pyserini) libraries.  
We test it on Python 3.8.


## Table of Contents
- [Setup](#setup)
- [Retrieval](#retrieval)
- [Evaluation](#evaluation)
  - [Language Models](#list-of-language-models)
  - [Evaluate models w/o retrieval](#evaluate-models-wo-retrieval)
  - [Evaluate models with retrieval](#evaluate-models-with-retrieval)
  - [Evaluate models with reranking](#reranking)
- [Citation](#citation)

## Setup

To install the required libraries in our repo, run:
```bash
pip install -r requirements.txt
```
To have a Pytorch version specific to your CUDA, [install](https://pytorch.org/) your version before running the above command.

## Retrieval

### BM25

Our BM25 preparation script works with Pyserini, so Java 11 is required - see their [installation guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md).  
If you have Java 11 installed, make sure your `JAVA_HOME` environment variable is set to the correct path. 
On a Linux system, the correct path might look something like `/usr/lib/jvm/java-11`.  
Then run:

```bash
python prepare_retrieval_data.py \
--retrieval_type sparse \
--tokenizer_name $MODEL_NAME \
--max_length 1024 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split [validation, test] \
--index_name wikipedia-dpr \
--forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
--stride 4 \
--output_file $RETRIEVAL_FILE \
--num_tokens_for_query 32 \
--num_docs 16 
```

### Dense Retrievers

We will soon add the code to reproduce our dense retrieval experiments. 

## Evaluation

### List of Language Models

In the paper, we give the results for the following models (replace `$MODEL_NAME` with one of those).  
Note that the larger models may need model parallelism (on a 40GB A100, we used model parallelism for OPT-30B and OPT-66B).  
See details below on how to apply this option.

* GPT-2: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
* GPT-Neo: `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B`, `EleutherAI/gpt-j-6B`
* OPT: `facebook/opt-125m`, `facebook/opt-350m`, `facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b`, `facebook/opt-13b`, `facebook/opt-30b`, `facebook/opt-66b`

### Evaluate models w/o retrieval

To run evaluation on models without retrieval, please use the following command (you can increase `stride` to 32 for faster evaluation):
```bash
python eval_lm.py \
--model_name $MODEL_NAME \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split [validation, test] \
--output_dir $OUTPUT_DIR \
--stride 4 \
--max_length 1024 \
[--model_parallelism]
```

### Evaluate models with retrieval:

To run models with retrieval, use the `$RETRIEVAL_FILE` output from the `prepare_retrieval_data.py` script:
```bash
python eval_lm.py \
--model_name $MODEL_NAME \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split [validation, test] \
--output_dir $OUTPUT_DIR \
--stride 4 \
--max_length 1024 \
[--model_parallelism] \
--retrieved_file $RETRIEVAL_FILE
```

Note: Our main retrieval flow assumes you want to use the top-scored passage from your retrieval file (`--ranking_strategy first`).

### Reranking 

Currently, we support `logprob` (the zero-shot method described in subsection 6.1) and `oracle` (to understand the potential gains from reranking).

For reranking, first you need to make sure you run the retrieval script with `num_docs=16` (or any other number you want to rerank on).
If you enable multiple GPUs, data parallelism will automatically be applied (each GPU will get different retrieved documents to condition on).
Then run:
```bash
python eval_lm.py \
--model_name $MODEL_NAME \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split [validation, test] \
--output_dir $OUTPUT_DIR \
--stride 4 \
--max_length 1024 \
[--model_parallelism] \
--retrieved_file $RETRIEVAL_FILE \
--ranking_strategy [logprob, oracle] \
--num_docs_to_rank 16 \
--ranking_logprob_past_tokens 16
```


## Citation

If you find our paper or code helpful, please cite our paper:
```
@article{ram2023ralm,
  author = {Ori Ram and Yoav Levine and Itay Dalmedigos and Dor Muhlgay and Amnon Shashua and Kevin Leyton-Brown and Yoav Shoham},
  title = {In-Context Retrieval-Augmented Language Models},
  year = {2023},
  url = {https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/63d9225812ceaf0fabb57767_In_Context_Retrieval_Augmented_Language_Models%20(1).pdf},
}
```
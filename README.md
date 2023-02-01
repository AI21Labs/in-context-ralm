# In-Context Retrieval-Augmented Language Models

This repo contains the code for reproducing the experiments from [AI21 Labs](https://www.ai21.com/)' paper [In-Context Retrieval-Augmented Language Models](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/63c6c20dec4479564db21819_NEW_In_Context_Retrieval_Augmented_Language_Models.pdf) (In-Context RALM).

Our code is mainly based on the [Transformers](https://github.com/huggingface/transformers/), [Pyserini](https://github.com/castorini/pyserini) and [Faiss](https://github.com/facebookresearch/faiss/) libraries.  
We test it on Python 3.8.


## Table of Contents
- [Setup](#setup)
- [Retrieval](#retrieval)
  - [BM25](#bm25)
  - [Dense Retrievers](#dense-retrievers)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Setup

To install the required libraries in our repo, run:
```bash
pip install -r requirements.txt
```

## Retrieval

### BM25

```bash
python prepare_retrieval_data.py \
--retrieval_type sparse \
--tokenizer_name $MODEL_NAME \
--max_length 1024 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split {validation, test} \
--index_name wikipedia-dpr \
--forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
--stride 4 \
--output_file $RETRIEVAL_FILE \
--num_tokens_for_query 32 \
--num_docs 16 
```

### Dense Retrievers

```bash
python prepare_retrieval_data.py \
--retrieval_type dense \
--tokenizer_name $MODEL_NAME \
--model_type 

--
```

## Evaluation

### Used Models

### Evaluate models w/o retrieval

To run evaluation on models without retrieval, please use the following command:
```bash
python eval_lm.py \
--model_name $MODEL_NAME \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split {validation, test} \
--output_dir $OUTPUT_DIR \
--stride 32 \
--max_length 1024 
```

### Evaluate models with retrieval:

To run models with retrieval, use the `$RETRIEVAL_FILE` output from the `prepare_retrieval_data.py` script:
```bash
python eval_lm.py \
--model_name $MODEL_NAME \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split {validation, test} \
--output_dir $OUTPUT_DIR \
--stride 4 \
--max_length 1024 \
--retrieved_file $RETRIEVAL_FILE
```

Note: Our main retrieval flow assumes you want to use the top-scored passage from your retrieval file (`--ranking_strategy first`).

#### Reranking 
For reranking, first you need to make sure you run the 


Note! Currently, we support only `oracle` and `zero-shot` (subsection 6.1 in the paper) reranking   
We still didn't add the code implementing our predictive reranking (subsection 6.2 in the paper).

## Citation

If you find our paper or code helpful, please cite our paper:
```
@article{ram2023ralm,
  author = {Ori Ram and Yoav Levine and Itay Dalmedigos and Dor Muhlgay and Amnon Shashua and Kevin Leyton-Brown and Yoav Shoham},
  journal = {arXiv preprint arXiv:????},
  title = {In-Context Retrieval-Augmented Language Models},
  year = {2023},
  url = {},
}
```
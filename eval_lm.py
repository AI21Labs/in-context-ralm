import os
import argparse
import json
import pickle

import numpy as np
import torch
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_dataset

from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer


def evaluate_logprob_with_retrieved_docs(
        model,
        tokenizer,
        device,
        encodings,
        begin_loc,
        end_loc,
        trg_len,
        retrieved_item,
        ranking_strategy,
        num_tokens_to_rank,
        retrieval_max_length,
        num_docs=-1
):
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

    if ranking_strategy == "first":
        assert num_docs in [-1, 1], f"In 'first' ranking strategy, unexpected number of docs to rank: {num_docs}"
        num_docs = 1
        chosen_doc_id = 0
    elif ranking_strategy == "random":
        chosen_doc_id = np.random.randint(num_docs)
        retrieved_item["retrieved_docs"] = [retrieved_item["retrieved_docs"][chosen_doc_id]]
        num_docs = 1

    num_docs_in_retrieved = len(retrieved_item["retrieved_docs"])
    num_docs = min(num_docs, num_docs_in_retrieved) if num_docs > 0 else num_docs_in_retrieved

    input_ids = input_ids.repeat(num_docs, 1)
    target_ids = input_ids.clone()
    labels_for_ranking = input_ids.clone()
    assert input_ids.size() == (num_docs, end_loc-begin_loc)

    for doc_id in range(num_docs):
        retrieved_example = retrieved_item["retrieved_docs"][doc_id]

        doc_title = retrieved_example["title"] if "title" in retrieved_example else None
        doc_text = retrieved_example["text"]
        if doc_title:
            doc_text = doc_title + "\n" + doc_text
        encoded_retrieved_text = tokenizer.encode(doc_text, max_length=retrieval_max_length, truncation=True)

        input_ids[doc_id, :len(encoded_retrieved_text)] = torch.tensor(encoded_retrieved_text, device=device)

    loss_fct = CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        lm_logits = model(input_ids).logits

        # Rank:
        if ranking_strategy in ["first", "random"]:
            batch_doc_id = 0
        else:
            if ranking_strategy == "oracle":
                labels_for_ranking[:, :-trg_len] = -100
                num_tokens_to_rank = trg_len  # We override this variable as it's not really relevant in oracle setting
            else:
                labels_for_ranking[:, :-trg_len-num_tokens_to_rank] = -100
                labels_for_ranking[:, -trg_len:] = -100
            labels_for_ranking = labels_for_ranking[:, 1:]
            assert torch.sum(labels_for_ranking[0] != -100).cpu().item() == num_tokens_to_rank

            lm_logits_for_ranking = lm_logits[..., :-1, :]
            ranking_loss = loss_fct(lm_logits_for_ranking.reshape(-1, lm_logits_for_ranking.size(-1)), labels_for_ranking.reshape(-1))
            ranking_loss = ranking_loss.view(num_docs, -1)
            per_doc_ranking_loss = torch.sum(ranking_loss, dim=-1)
            chosen_doc_id = torch.argmin(per_doc_ranking_loss).cpu().item()
            batch_doc_id = chosen_doc_id

        # Calculate logprob of the chosen doc:
        lm_logits = lm_logits[batch_doc_id, -trg_len-1:-1, :]
        labels = target_ids[batch_doc_id, -trg_len:]
        loss = loss_fct(lm_logits, labels)
        token_ppls = loss.cpu()
        tokens_to_predict = labels.view(-1).cpu().tolist()
        loss = token_ppls.sum()

    return loss, chosen_doc_id, token_ppls.tolist(), tokens_to_predict


def eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length,
        output_dir=None,
        stride=4,
        normalization_level="word",
        retrieval_dataset=None,
        retrieval_max_length=256,
        ranking_strategy="first",
        num_docs_to_rank=1,
        num_tokens_to_rank_logprob=16
):
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")

    print("Max context length:", max_length)
    # Number of tokens in dataset
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)

    if normalization_level == "word":
        counter = dataset.count(" ")
    elif normalization_level == "token":
        counter = dataset_len
    else:
        raise ValueError(f"Unknown normalization_level: '{normalization_level}'")

    print("Normalization factor (num tokens/words..):", counter)

    nlls = []
    prev_end_loc = 0

    idx = 0
    all_token_ppls = []
    all_tokens_to_predict = []
    all_chosen_doc_ids = [None]
    num_inputs_no_retrieval = 0
    for begin_loc in tqdm(range(0, dataset_len, stride)):
        end_loc = min(begin_loc + max_length, dataset_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        if idx > 0 and retrieval_dataset is not None and len(retrieval_dataset[idx]["retrieved_docs"]) > 0:
            retrieved_example = retrieval_dataset[idx]
            assert retrieved_example["begin_location"] == prev_end_loc
            assert retrieved_example["end_location"] == end_loc

            neg_log_likelihood, chosen_doc_id, token_ppls, tokens_to_predict = evaluate_logprob_with_retrieved_docs(
                model, tokenizer, device, encodings, begin_loc, end_loc, trg_len, retrieved_example,
                ranking_strategy=ranking_strategy,
                num_tokens_to_rank=num_tokens_to_rank_logprob,
                retrieval_max_length=retrieval_max_length,
                num_docs=num_docs_to_rank
            )
            all_chosen_doc_ids.append(chosen_doc_id)
        else:
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # Calculate per-token loss
                if trg_len < max_length:
                    neg_log_likelihood = outputs.loss * trg_len
                    lm_logits = outputs.logits[..., -trg_len-1:-1, :]
                    labels = target_ids[..., -trg_len:]
                else:
                    neg_log_likelihood = outputs.loss * (max_length - 1)
                    lm_logits = outputs.logits[..., :-1, :]
                    labels = target_ids[..., 1:]
                neg_log_likelihood = neg_log_likelihood.to(torch.float32).squeeze().cpu()
                lm_logits = lm_logits.to(torch.float32)

                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).cpu()
                token_ppls = loss.tolist()
                tokens_to_predict = labels.view(-1).cpu().tolist()

        nlls.append(neg_log_likelihood)
        all_token_ppls.append(token_ppls)
        all_tokens_to_predict.append(tokens_to_predict)
        assert len(all_token_ppls) == len(all_tokens_to_predict)

        prev_end_loc = end_loc
        idx += 1
        if end_loc == dataset_len:
            break

    assert retrieval_dataset is None or len(retrieval_dataset) == idx

    ppl = torch.exp(torch.stack(nlls).sum() / counter).item()
    print("Perplexity:", ppl)
    ppl_to_assert = np.exp(sum([sum(x) for x in all_token_ppls]) / counter)
    assert np.abs(ppl - ppl_to_assert) < 1e-3, f"{ppl:.3f}, {ppl_to_assert:.3f}"

    if output_dir is not None:
        d = {"eval_perplexity": ppl}
        if retrieval_dataset is not None:
            d["num_input_no_retrieval"] = num_inputs_no_retrieval
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")

        with open(os.path.join(output_dir, "ppls.pkl"), "wb") as f:
            to_dump = (all_token_ppls, all_tokens_to_predict, all_chosen_doc_ids) if all_chosen_doc_ids \
                else (all_token_ppls, all_tokens_to_predict)
            pickle.dump(to_dump, f)


def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir)
    print_args(args, output_dir=args.output_dir)

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length

    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()
    retrieval_dataset = None
    if args.retrieved_file is not None:
        with open(args.retrieved_file, "r") as f:
            retrieval_dataset = json.load(f)

    eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length=max_length,
        output_dir=args.output_dir,
        stride=args.stride,
        normalization_level=args.normalization_level,
        retrieval_dataset=retrieval_dataset,
        retrieval_max_length=args.retrieved_max_length,
        ranking_strategy=args.ranking_strategy,
        num_docs_to_rank=args.num_docs_to_rank,
        num_tokens_to_rank_logprob=args.ranking_logprob_past_tokens,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--normalization_level", choices=["word", "token"], default="word")

    # retrieval params
    parser.add_argument("--retrieved_file", type=str, default=None)
    parser.add_argument("--retrieved_max_length", type=int, default=256)
    parser.add_argument("--ranking_strategy", type=str, choices=["first", "logprob", "oracle", "random"], default="first")
    parser.add_argument("--num_docs_to_rank", type=int, default=-1)
    parser.add_argument("--ranking_logprob_past_tokens", type=int, default=16)

    args = parser.parse_args()

    main(args)

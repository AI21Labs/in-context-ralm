import json
import sys
import argparse

import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ralm.file_utils import print_args
from ralm.retrievers.retrieval_factory import add_retriever_args, get_retriever

RETRIEVAL_TYPES = [
    "dense",
    "sparse",
]


def main(args):
    # Dump args
    print_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Loading dataset...")
    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)

    print(f"Creating retriever of type {args.retrieval_type}...")
    retriever = get_retriever(args.retrieval_type, args, tokenizer)

    prev_end_loc = 0
    data = []
    for begin_loc in tqdm(range(0, dataset_len, args.stride)):
        end_loc = min(begin_loc + args.max_length, dataset_len)
        target_begin_loc = prev_end_loc

        # d = retriever.retrieve(encodings.input_ids, target_begin_loc, end_loc, title=None)

        d = {
            "begin_location": target_begin_loc,
            "end_location": end_loc,
            "future": tokenizer.decode(encodings.input_ids[0, target_begin_loc:end_loc])
        }

        data.append(d)
        prev_end_loc = end_loc

        if end_loc >= dataset_len:
            break

    batch_size = 1000
    for i in range(0, len(data), batch_size):
        if i > 0:
            print(f"Finished processing {i}/{len(data)} strides")
        retriever.retrieve(encodings.input_ids, data[i:i+batch_size], k=args.num_docs)

    print(f"Finished processing {len(data)}/{len(data)} strides")
    print(f"Writing to {args.output_file}")
    with open(args.output_file, "w") as f:
        f.write(json.dumps(data, indent=4))
        f.write("\n")

    print("Done!")


if __name__ == '__main__':
    assert sys.argv[1] == "--retrieval_type"
    retrieval_type = sys.argv[2]

    assert retrieval_type in RETRIEVAL_TYPES

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", required=True, type=str)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")

    # Model params
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=32)

    # Retrieval params
    parser.add_argument("--retrieval_type", required=True, choices=RETRIEVAL_TYPES)
    parser.add_argument("--num_docs", type=int, default=1)
    add_retriever_args(parser, retrieval_type)

    args = parser.parse_args()
    main(args)

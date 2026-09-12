"""Export only actor weights from a recovery checkpoint, without a GPU."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    args = parser.parse_args()
    if (args.output / "export.json").exists():
        saved = json.loads((args.output / "export.json").read_text())
        if saved["checkpoint"] != str(args.checkpoint.resolve()):
            raise ValueError("Output already belongs to a different checkpoint")
        print(f"Already exported: {args.output}")
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete or unrelated output: {args.output}")
    torch.set_num_threads(4)
    reader = dcp.FileSystemReader(args.checkpoint)
    metadata = reader.read_metadata()
    prefix = "worker0.model."
    weights = {key[len(prefix):]: torch.empty(value.size, dtype=value.properties.dtype)
               for key, value in metadata.state_dict_metadata.items() if key.startswith(prefix)}
    dcp.load({"worker0": {"model": weights}}, storage_reader=reader)
    cfg = AutoConfig.from_pretrained(args.model)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    temporary = args.output.with_name(args.output.name + ".exporting")
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temporary, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.model).save_pretrained(temporary)
    (temporary / "export.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "model": args.model,
        "kind": "actor_weights_only", "training_recovery": False,
    }, indent=2))
    os.replace(temporary, args.output)
    print(f"Exported actor weights to {args.output}", flush=True)


if __name__ == "__main__":
    main()

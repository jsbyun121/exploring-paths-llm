from omegaconf import OmegaConf
import os
import asyncio
import importlib
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
from RL2.workers import Worker
from RL2.datasets import get_tensor_dict, pack_tensor_dicts
from RL2.utils.comm import split_and_scatter_list, gather_and_concat_list
from RL2.utils.logging import time_logger, gather_and_log


class Rollout(Worker):

    def __init__(self, config):
        super().__init__(config, None)
        
        self.prepare_environment_variables()
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.prepare_environment()

            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.llm = Engine(
                model_path=config.model_name,
                dtype=config.dtype,
                tp_size=self.device_mesh["tp"].size(),
                mem_fraction_static=config.gpu_memory_utilization,
                mamba_scheduler_strategy=getattr(
                    config, "mamba_scheduler_strategy", "auto"
                ),
                enable_memory_saver=getattr(
                    config, "release_memory_for_training", True
                ),
                port=config.base_port + dist.get_rank()
            )
        
            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

            # Engine owns the event loop used by its tokenizer-manager
            # communicators. Generation and synchronous control operations
            # (weight updates, cache flushes, sleep/wake) must share that loop;
            # replacing it after Engine initialization can strand communicator
            # tasks on a loop that is no longer being driven.
            self.event_loop = self.llm.loop
            asyncio.set_event_loop(self.event_loop)

        dist.barrier()

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % self.config.tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {self.config.tp_size}."
        self.dp_size = world_size // self.config.tp_size
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(self.dp_size, self.config.tp_size)
        )

    def prepare_environment_variables(self):

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            cuda_visible_devices = cuda_visible_devices.split(",")
            cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
        else:
            cuda_visible_device = os.environ["LOCAL_RANK"]
        cuda_visible_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            cuda_visible_device,
            self.device_mesh["tp"].get_group(),
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    def prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    def initialize_state_dict(self, state_text):

        states = self.tokenizer.encode(state_text, add_special_tokens=False)
        return {
            "states": states,
            "actions": len(states) * [0],
            "action_mask": len(states) * [0],
            "logps": len(states) * [0],
            "rewards": len(states) * [0]
        }
    
    def get_tensor_dict(self, state_dict):

        tensor_dict = get_tensor_dict(
            state_dict["states"],
            state_dict["actions"],
            state_dict["action_mask"]
        )
        tensor_dict["llm_logps"] = torch.FloatTensor(state_dict["logps"][1:])
        tensor_dict["rewards"] = torch.FloatTensor(state_dict["rewards"][1:])
        return tensor_dict
        
    async def rollout(self, ex, train):

        state_text = (
            ex["prompt"] if "prompt" in ex else
            await self.env.reset(ex["extra_info"])
        )
        state_dict = self.initialize_state_dict(state_text)
        env_response = {"extra_info": ex["extra_info"]}
        tensor_dicts = []
        metric = defaultdict(list)
        scores = []
        for turn in range(1, self.config.max_turns + 1):

            sampling_params = dict(self.train_sampling_params if train else self.test_sampling_params)
            if "_sampling_seed" in ex:
                sampling_params["sampling_seed"] = (ex["_sampling_seed"] + turn - 1) % (2**31)
            llm_response = await self.llm.async_generate(
                input_ids=state_dict["states"],
                sampling_params=sampling_params,
                return_logprob=True
            )

            action_text = llm_response["text"]
            env_response = await self.env.step(
                state_text, action_text, env_response["extra_info"]
            )

            meta_info = llm_response["meta_info"]
            logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
            state_dict["states"].extend(action)
            state_dict["actions"].extend(action)
            state_dict["action_mask"].extend(len(action) * [1])
            state_dict["logps"].extend(logp)
            state_dict["rewards"].extend((len(action) - 1) * [0] + [env_response["reward"]])
            metric["response_length"].append(meta_info["completion_tokens"])
            metric["length_clip_ratio"].append(
                meta_info["finish_reason"]["type"] == "length"
            )
            scores.append(env_response["score"])

            if turn == self.config.max_turns or env_response["done"]:
                tensor_dicts.append(self.get_tensor_dict(state_dict))
                break
            if env_response["next_state"].startswith(state_text + action_text):
                state_dict_delta = self.initialize_state_dict(
                    env_response["next_state"][len(state_text + action_text):]
                )
                for k, v in state_dict_delta.items():
                    state_dict[k].extend(v)
            else:
                tensor_dicts.append(self.get_tensor_dict(state_dict))
                state_dict = self.initialize_state_dict(env_response["next_state"])
            state_text = env_response["next_state"]

        metric["n_turns"].append(turn)
        metric["scores"].append(sum(scores))

        return tensor_dicts, metric

    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):

        # The data is distributed from rank 0 before each worker operation
        # and gathered before the next operation, which facilitates to do
        # model-agnostic operations, e.g., computing advantages, globally 
        # and guarantees the load balancing across all model computations.
        if self.device_mesh["tp"].get_local_rank() == 0:

            if data_list is not None and getattr(self.config, "sampling_seed_base", None) is not None:
                data_list = [{**ex, "_sampling_seed": (
                    self.config.sampling_seed_base + (step if train else 0) * 1000003 + index
                ) % (2**31)} for index, ex in enumerate(data_list)]
            data_list = split_and_scatter_list(
                data_list, self.device_mesh["dp"]
            )
            outputs = self.event_loop.run_until_complete(
                tqdm.gather(
                    *(self.rollout(ex, train) for ex in data_list),
                    desc="Rollout",
                    position=1,
                    leave=False,
                    disable=(dist.get_rank() != 0)
                )
            )
            if train and getattr(
                self.config, "release_memory_for_training", True
            ):
                # If test, llm will soon be called again. See `Trainer.train`.
                self.llm.release_memory_occupation()

        dist.barrier()

        if self.device_mesh["tp"].get_local_rank() == 0:

            all_tensor_dicts, metrics = map(list, zip(*outputs))

            suffix = "train" if train else "test"
            metrics = {
                f"{k}/{suffix}": sum([metric[k] for metric in metrics], [])
                for k in metrics[0].keys()
            }
            self.last_metrics = gather_and_log(metrics, self.device_mesh["dp"], step)

            if not train:
                return

            all_tensor_dicts = gather_and_concat_list(
                all_tensor_dicts, self.device_mesh["dp"]
            )

            if dist.get_rank() == 0:

                group_size = self.config.responses_per_prompt
                if group_size > 1 and self.config.dynamic_filtering:

                    rewards = torch.FloatTensor([
                        sum([td["rewards"].sum().item() for td in tensor_dicts])
                        for tensor_dicts in all_tensor_dicts
                    ]).view(-1, group_size)
                    are_filtered = rewards.std(-1) == 0
                    all_tensor_dicts = sum([
                        all_tensor_dicts[idx * group_size:(idx + 1) * group_size]
                        for idx, is_filtered in enumerate(are_filtered)
                        if not is_filtered
                    ], [])
                    wandb.log({
                        "dynamic_filtering_ratio": are_filtered.float().mean().item()
                    }, step=step)

                tensor_dicts = sum(all_tensor_dicts, [])
                tensor_dict = pack_tensor_dicts(tensor_dicts)
                seqs = torch.LongTensor([
                    len(tensor_dicts) for tensor_dicts in all_tensor_dicts
                ])
                cu_seqs = torch.cumsum(
                    torch.cat((torch.LongTensor([0]), seqs)), dim=0
                )
                
                return tensor_dict, cu_seqs

        return None, None
        
    @time_logger("update_rollout")
    def update(self, actor, step):

        torch.cuda.empty_cache()
        dist.barrier()
        # or llm.resume_memory_occupation() may OOM
        if (
            self.device_mesh["tp"].get_local_rank() == 0
            and getattr(self.config, "release_memory_for_training", True)
        ):
            self.llm.resume_memory_occupation()
        
        tp_size = self.device_mesh["tp"].size()
        tp_rank = self.device_mesh["tp"].get_local_rank()
        named_tensors = []
        for name, tensor in actor.state_dict.items():
            # get_state_dict(..., cpu_offload=True) already gives us CPU
            # tensors. Moving them back to CUDA before serialization makes
            # PyTorch use CUDA IPC; that path requires pidfd_getfd, which is
            # commonly blocked by container seccomp profiles. Serialize CPU
            # storage instead and let SGLang's _unwrap_tensor move it to the
            # inference device after deserialization.
            tensor = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            tensor = tensor.detach().to("cpu").contiguous()

            # With TP=1, Engine can serialize the CPU tensor directly. Avoid
            # wrapping an already-serialized storage descriptor inside a
            # second serialized payload; rebuilding that nested descriptor can
            # deadlock in restricted containers. LocalSerializedTensor is only
            # needed to select among shards contributed by multiple TP ranks.
            if tp_size == 1:
                if tp_rank == 0:
                    named_tensors.append((name, tensor))
                continue

            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor
            )
            serialized_tensors = [
                None for _ in range(tp_size)
            ] if tp_rank == 0 else None
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            if tp_rank == 0:
                named_tensors.append(
                    (name, LocalSerializedTensor(values=serialized_tensors))
                )

        if tp_rank == 0:
            print(f"Updating rollout model with {len(named_tensors)} tensors...")
            self.llm.update_weights_from_tensor(
                named_tensors=named_tensors,
                flush_cache=True,
            )
            print("Rollout model weights updated.")
        actor.state_dict.clear()
        dist.barrier()

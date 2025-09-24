import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler
from tqdm import tqdm
from .scorer import Llama_Scorer

class InfoBatch(Dataset):
    def __init__(self, dataset, data_method="random", data_ratio=0.5):
        self.dataset = dataset
        self.method = data_method
        self.ratio = data_ratio
        self.scores = np.ones([len(self.dataset)])
        self.complexities = np.zeros([len(self.dataset)])
        self.qualities = np.zeros([len(self.dataset)])
        self.embeds = np.ones((len(self.dataset), 1, 4096))
        self.grand = np.zeros([len(self.dataset)])
        self.el2n = np.zeros([len(self.dataset)])
        # self.complexities = np.load('/home/tiger/Data_Token_Pruning/llama_complexity.npy')
        # self.qualities = np.load('/home/tiger/Data_Token_Pruning/llama_quality.npy')
        # self.embeds = np.load('/home/tiger/Data_Token_Pruning/llama_embed.npy')
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0
        self.seq = list(range(len(self.dataset)))
        # self.filter = Combined_Filter(threshold=0.9, data_size=int(self.ratio*len(self.dataset)), sort_key="complexity_scores,quality_scores", chunk_size=100000, distance_metric="cosine", embedding_field="embedding")
        # debug flags (only print once per method on rank0)
        self.debug_longest_printed = False
        self.debug_entropy_printed = False

    def __setscore__(self, indices, values):
        count = torch.zeros(len(self.dataset), device="cuda")
        delta_scores = torch.zeros(len(self.dataset), device="cuda")
        count[indices] = 1
        delta_scores[indices] = torch.tensor(values, device="cuda")
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(delta_scores, op=dist.ReduceOp.SUM)
        delta_scores /= torch.clamp(count, min=1)
        valid_mask = (delta_scores != 0).cpu()
        self.scores[valid_mask] = delta_scores[valid_mask].cpu().numpy()
    
    def __setcomplexity__(self, indices, values):
        count = torch.zeros(len(self.dataset), device="cuda")
        delta_scores = torch.zeros(len(self.dataset), device="cuda")
        count[indices] = 1
        delta_scores[indices] = torch.tensor(values, device="cuda")
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(delta_scores, op=dist.ReduceOp.SUM)
        delta_scores /= torch.clamp(count, min=1)
        valid_mask = (delta_scores != 0).cpu()
        self.complexities[valid_mask] = delta_scores[valid_mask].cpu().numpy()
    
    def __setquality__(self, indices, values):
        count = torch.zeros(len(self.dataset), device="cuda")
        delta_scores = torch.zeros(len(self.dataset), device="cuda")
        count[indices] = 1
        delta_scores[indices] = torch.tensor(values, device="cuda")
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(delta_scores, op=dist.ReduceOp.SUM)
        delta_scores /= torch.clamp(count, min=1)
        valid_mask = (delta_scores != 0).cpu()
        self.qualities[valid_mask] = delta_scores[valid_mask].cpu().numpy()

    def __setembed__(self, indices, values):
        count = torch.zeros(len(self.dataset), device="cuda")
        delta_scores = torch.zeros(((len(self.dataset), 1, 4096)), device="cuda")
        count[indices] = 1
        delta_scores[indices] = torch.tensor(values, device="cuda")
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(delta_scores, op=dist.ReduceOp.SUM)
        idx = torch.nonzero(count > 1).squeeze(1)
        if len(idx) > 0:
            delta_scores[idx] = delta_scores[idx] / count[idx][:, None, None]
        valid_mask = (count != 0).cpu()
        self.embeds[valid_mask] = delta_scores[valid_mask].cpu().numpy()

    def __setgrand__(self, indices, values):
        count = torch.zeros(len(self.dataset), device="cuda")
        delta_scores = torch.zeros(len(self.dataset), device="cuda")
        count[indices] = 1
        delta_scores[indices] = torch.tensor(values, device="cuda")
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(delta_scores, op=dist.ReduceOp.SUM)
        delta_scores /= torch.clamp(count, min=1)
        valid_mask = (delta_scores != 0).cpu()
        self.grand[valid_mask] = delta_scores[valid_mask].cpu().numpy()

    def __setel2n__(self, indices, values):
        count = torch.zeros(len(self.dataset), device="cuda")
        delta_scores = torch.zeros(len(self.dataset), device="cuda")
        count[indices] = 1
        delta_scores[indices] = torch.tensor(values, device="cuda")
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(delta_scores, op=dist.ReduceOp.SUM)
        delta_scores /= torch.clamp(count, min=1)
        valid_mask = (delta_scores != 0).cpu()
        self.el2n[valid_mask] = delta_scores[valid_mask].cpu().numpy()

    def __setmodel__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        i_list = i.item() if isinstance(i, np.integer) else i
        return dict(input_ids=self.dataset[i_list]["input_ids"], labels=self.dataset[i_list]["labels"], text=self.dataset[i_list]["text"], indices=i, weights=self.weights[i])

    def prune(self):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance
        
        pruned_samples = []

        if self.method == "wise":
            # No-op data method: do not prune at dataset/sampler level.
            samples = list(range(len(self.dataset)))
            pruned_samples.extend(samples)
        elif self.method == "random":
            samples = list(range(len(self.dataset)))
            selected = np.random.choice(samples, int(self.ratio*len(samples)),replace=False)
            pruned_samples.extend(selected)
        elif self.method == "longest":
            # 按指令长度从低到高排序，保留前 ratio 比例（最短的前 ratio）
            try:
                instructions = [
                    (x.get("text", {}) or {}).get("instruction", "") if isinstance(x, dict) else ""
                    for x in self.dataset
                ]
            except Exception:
                instructions = [""] * len(self.dataset)

            lengths = np.array([len(s) for s in instructions], dtype=np.int64)
            sorted_indices = np.argsort(lengths)  # 升序（短→长）
            keep_num = max(1, int(self.ratio * len(self.dataset)))
            keep_num = min(keep_num, len(self.dataset))
            # debug summary (rank0 only, print once)
            try:
                if (not self.debug_longest_printed) and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
                    empty_cnt = int(np.sum(lengths == 0))
                    if lengths.size > 0:
                        min_len = int(lengths.min())
                        mean_len = float(lengths.mean())
                        max_len = int(lengths.max())
                    else:
                        min_len = 0
                        mean_len = 0.0
                        max_len = 0
                    print(f"[DataMethod][longest] total={len(self.dataset)} empty_instr={empty_cnt} keep_num={keep_num} len(min/mean/max)={min_len}/{mean_len:.2f}/{max_len}")
                    preview = sorted_indices[:min(5, keep_num)].tolist()
                    for idx in preview:
                        instr = instructions[idx] if idx < len(instructions) else ""
                        instr_preview = instr[:80].replace("\n", " ")
                        print(f"[DataMethod][longest] sample idx={idx} len={lengths[idx]} instr_preview={instr_preview}")
                    self.debug_longest_printed = True
            except Exception:
                pass
            pruned_samples.extend(sorted_indices[:keep_num].tolist())
        elif self.method == "entropy":
            # 按指令熵从低到高排序，保留前 ratio 比例（熵低的前 ratio）
            try:
                instructions = [
                    (x.get("text", {}) or {}).get("instruction", "") if isinstance(x, dict) else ""
                    for x in self.dataset
                ]
            except Exception:
                instructions = [""] * len(self.dataset)

            def str_entropy(s: str) -> float:
                if not s:
                    return 0.0
                # 字符级香农熵（自然对数底）
                values, counts = np.unique(list(s), return_counts=True)
                probs = counts.astype(np.float64) / counts.sum()
                return float(-(probs * np.log(probs + 1e-12)).sum())

            ent = np.array([str_entropy(s) for s in instructions], dtype=np.float64)
            sorted_indices = np.argsort(ent)  # 升序（低熵→高熵）
            keep_num = max(1, int(self.ratio * len(self.dataset)))
            keep_num = min(keep_num, len(self.dataset))
            # debug summary (rank0 only, print once)
            try:
                if (not self.debug_entropy_printed) and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
                    empty_cnt = int(np.sum([len(s) == 0 for s in instructions]))
                    if ent.size > 0:
                        min_ent = float(ent.min())
                        mean_ent = float(ent.mean())
                        max_ent = float(ent.max())
                    else:
                        min_ent = 0.0
                        mean_ent = 0.0
                        max_ent = 0.0
                    print(f"[DataMethod][entropy] total={len(self.dataset)} empty_instr={empty_cnt} keep_num={keep_num} entropy(min/mean/max)={min_ent:.4f}/{mean_ent:.4f}/{max_ent:.4f}")
                    preview = sorted_indices[:min(5, keep_num)].tolist()
                    for idx in preview:
                        instr = instructions[idx] if idx < len(instructions) else ""
                        instr_preview = instr[:80].replace("\n", " ")
                        print(f"[DataMethod][entropy] sample idx={idx} H={ent[idx]:.4f} instr_preview={instr_preview}")
                    self.debug_entropy_printed = True
            except Exception:
                pass
            pruned_samples.extend(sorted_indices[:keep_num].tolist())
        elif self.method == "infobatch":
            data_size = int(self.ratio*len(self.dataset))
            b = self.scores<self.scores.mean()
            well_learned_samples = np.where(b)[0]
            data_size = int(self.ratio*len(self.dataset))
            pruned_samples.extend(np.where(np.invert(b))[0])
            selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)
            self.reset_weights()
            if len(selected)>0:
                self.weights[selected]=1/self.ratio
                pruned_samples.extend(selected)
            if len(pruned_samples) != len(self.dataset):
                pruned_samples = pruned_samples[:data_size]
        # elif self.method == "deita" or self.method == "deita-grand" or self.method == "deita-el2n":
        elif "deita" in self.method:
            if np.all(self.embeds == 1):
                pruned_samples = list(range(len(self.dataset)))
            else: 
                # embeddings_flat = self.embeds.squeeze(1)
                # df_data = pd.DataFrame({
                #     "embedding": [e for e in embeddings_flat], 
                #     "complexity_scores": self.complexities,
                #     "quality_scores": self.qualities,
                # })
                # if "idx" not in df_data.columns:
                #     df_data["idx"] = df_data.index
                # pruned_samples = self.filter.filter(df_data)["idx"].tolist()
                # np.save("./llama_complexity.npy", self.complexities)
                # np.save("./llama_quality.npy", self.qualities)
                # np.save("./llama_embed.npy", self.embeds)
                # print(f"Saved embed to ./llama_embed.npy")

                auto_scorer = Llama_Scorer(self.model, self.tokenizer)
                complexities = []
                qualities = []
                instructions = np.array([x["text"]["instruction"] for x in self.dataset])
                outputs = np.array([x["text"]["output"] for x in self.dataset])
                indices = list(range(dist.get_rank(), len(self.dataset), dist.get_world_size()))
                batch = 64
                for i in tqdm(range(0, len(indices), batch)):
                # for item in tqdm(self.dataset[dist.get_rank()::dist.get_world_size()]["text"]):
                    complexity = auto_scorer.infer_complexity(instructions[indices[i:i+batch]])
                    quality = auto_scorer.infer_quality(instructions[indices[i:i+batch]], outputs[indices[i:i+batch]])
                    complexities.extend(complexity)
                    qualities.extend(quality)
                self.__setcomplexity__(indices, np.array(complexities, dtype=np.float32))
                self.__setquality__(indices, np.array(qualities, dtype=np.float32))
                        
                threshold = 0.9
                data_size = int(self.ratio*len(self.dataset))
                if self.method == "deita2":
                    scores = self.complexities * self.qualities
                elif self.method == "deita-grand2":
                    alpha = 1
                    beta = 1
                    gamma = 1
                    scores = alpha*self.complexities + beta*self.qualities + gamma*self.grand
                elif self.method == "deita-el2n2":
                    alpha = 1
                    beta = 1
                    gamma = 1
                    scores = alpha*self.complexities + beta*self.qualities + gamma*self.el2n
                sorted_indices = np.argsort(-scores)
                embed = torch.tensor(self.embeds)
                embed = embed.squeeze(1)
                embed = embed[sorted_indices]
                embed = F.normalize(embed, dim=1)
                sim_matrix = torch.matmul(embed, embed.T)
                tri_mask = torch.tril(torch.ones_like(sim_matrix), diagonal=-1).bool()
                pruned_samples.append(sorted_indices[0])

                for i in tqdm(range(1, embed.shape[0])):
                    sim = sim_matrix[i]
                    mask = tri_mask[i]
                    if torch.all(sim[mask] <= threshold):
                        pruned_samples.append(sorted_indices[i])
                    else:
                        tri_mask[i, :] = False
                        tri_mask[:, i] = False
                    
                    if len(pruned_samples) >= data_size:
                        break

                if len(pruned_samples) >= data_size:
                    pruned_samples = pruned_samples[:data_size]
                else:
                    lack_num = data_size - len(pruned_samples)
                    remaining_indices = sorted_indices[~np.isin(sorted_indices, pruned_samples)]
                    pruned_samples.extend(remaining_indices[:lack_num])

        print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
        self.save_num += len(self.dataset)-len(pruned_samples)
        # np.random.shuffle(pruned_samples)
        self.seq = pruned_samples
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler(Sampler):
    def __init__(self, infobatch_dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.infobatch_dataset = infobatch_dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seq = None
        self.seed = 0
        self.reset()

    def reset(self):
        # np.random.seed(self.seed)
        # self.seed+=1
        self.seq = self.infobatch_dataset.prune()
        self.new_length = len(self.seq)

        self.num_samples = int(math.ceil(self.new_length / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.seq += self.seq[:(self.total_size - len(self.seq))]  
        self.seq = self.seq[self.rank:self.total_size:self.num_replicas]
        np.random.shuffle(self.seq)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self
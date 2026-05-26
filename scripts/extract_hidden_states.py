import datasets
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import transformers

from beartype import beartype
import hydra
from omegaconf import DictConfig

from src.datasets.speech_equivalence import SpeechHiddenStateDataset
from src.models.transformer import prepare_processor


# ---------------------------------------------------------------------------
# Thin Dataset wrappers used by DataLoader
# ---------------------------------------------------------------------------


class _LengthSortedDataset(Dataset):
    """Wraps a HuggingFace dataset and exposes items by length-sorted index."""

    def __init__(self, hf_dataset: datasets.Dataset, sorted_indices: list[int]):
        self._ds = hf_dataset
        self._sorted_indices = sorted_indices

    def __len__(self) -> int:
        return len(self._sorted_indices)

    def __getitem__(self, pos: int) -> dict:
        orig_idx = self._sorted_indices[pos]
        item = self._ds[orig_idx]
        return {"input_values": item["input_values"], "orig_idx": orig_idx}


class _QueryDataset(Dataset):
    """Dataset of (item_idx, t, keypoint) pseudo-causal queries.

    Each query represents one output frame to extract from the model.
    ``queries`` must already be sorted by keypoint value before being passed
    here so that the DataLoader's sequential sampler naturally groups similar-
    length clips together.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        queries: list[tuple[int, int, int]],
    ):
        self._ds = hf_dataset
        self._queries = queries

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, pos: int) -> dict:
        item_idx, t, keypoint = self._queries[pos]
        audio = torch.as_tensor(self._ds[item_idx]["input_values"])
        # Clip to keypoint and zero out beyond (matches original masking)
        audio_clip = audio[:keypoint].clone()
        # output frame index (0-based in the model output sequence)
        frame_pos = t - 1
        return {
            "audio_clip": audio_clip,   # shape: (keypoint,)
            "keypoint": keypoint,
            "item_idx": item_idx,
            "frame_pos": frame_pos,
        }


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------


def _collate_npc(batch: list[dict], processor: transformers.Wav2Vec2Processor) -> dict:
    """Collate a batch for the non-pseudo-causal path."""
    padded = processor.pad(
        [{"input_values": item["input_values"]} for item in batch],
        max_length=None,
        return_tensors="pt",
        return_attention_mask=True,
    )
    orig_idxs = [item["orig_idx"] for item in batch]
    padded["orig_idxs"] = orig_idxs
    return padded


def _collate_pc(batch: list[dict]) -> dict:
    """Collate a batch of pseudo-causal queries.

    Pads all audio clips to the length of the longest clip in the batch,
    builds attention masks, and stacks metadata tensors.
    """
    max_len = max(item["keypoint"] for item in batch)
    batch_size = len(batch)

    input_values = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.int32)

    item_idxs = []
    frame_positions = []

    for j, item in enumerate(batch):
        kp = item["keypoint"]
        input_values[j, :kp] = item["audio_clip"]
        attention_mask[j, :kp] = 1
        item_idxs.append(item["item_idx"])
        frame_positions.append(item["frame_pos"])

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "item_idxs": item_idxs,
        "frame_positions": frame_positions,
    }


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


@beartype
def extract_hidden_states(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    processor: transformers.Wav2Vec2Processor,
    layer: int,
    pseudo_causal: bool = False,
    batch_size: int = 32,
) -> SpeechHiddenStateDataset:
    flat_idxs: list[tuple[int, int]] = []
    frame_states_list: list[torch.Tensor] = []
    compression_ratios: dict[int, float] = {}

    model.eval()

    device = model.device
    use_cuda = device.type == "cuda"

    num_items = len(dataset)

    if not pseudo_causal:
        # ------------------------------------------------------------------
        # Non-pseudo-causal path: batch over items, length-bucketed
        # ------------------------------------------------------------------

        # Sort items by audio length so that batches have minimal padding waste
        audio_lengths = [
            len(dataset[i]["input_values"]) for i in range(num_items)
        ]
        sorted_indices = sorted(range(num_items), key=lambda i: audio_lengths[i])

        length_sorted_ds = _LengthSortedDataset(dataset, sorted_indices)

        def collate_fn(batch):
            return _collate_npc(batch, processor)

        loader = DataLoader(
            length_sorted_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_cuda,
            collate_fn=collate_fn,
        )

        # Accumulate per-item results so we can emit in item-index order later
        per_item_states: dict[int, torch.Tensor] = {}

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Extracting hidden states"):
                orig_idxs = batch.pop("orig_idxs")
                input_values = batch["input_values"].to(device, non_blocking=use_cuda)
                attention_mask = batch["attention_mask"].to(device, non_blocking=use_cuda)

                output = model(
                    output_hidden_states=True,
                    input_values=input_values,
                    attention_mask=attention_mask,
                )

                input_lengths = batch["attention_mask"].sum(dim=1)
                frame_lengths = model._get_feat_extract_output_lengths(input_lengths)

                batch_hidden_states = output.hidden_states[layer].cpu()
                batch_compression_ratios = frame_lengths / input_lengths.to(torch.float32)

                for idx, num_frames_i, hidden_states_i, compression_i in zip(
                    orig_idxs, frame_lengths, batch_hidden_states, batch_compression_ratios
                ):
                    idx = int(idx)
                    num_frames_i = int(num_frames_i)
                    # .clone() avoids holding a view into the padded batch tensor
                    per_item_states[idx] = hidden_states_i[:num_frames_i].clone()
                    compression_ratios[idx] = compression_i.item()

        # Emit in original item order (0 .. num_items-1)
        for idx in range(num_items):
            states_i = per_item_states[idx]
            num_frames_i = states_i.shape[0]
            flat_idxs.extend([(idx, j) for j in range(num_frames_i)])
            frame_states_list.append(states_i)

    else:
        # ------------------------------------------------------------------
        # Pseudo-causal path: cross-item batching over (item, t, keypoint)
        # queries, sorted by keypoint length
        # ------------------------------------------------------------------

        max_length = max(
            pa.compute.list_value_length(dataset._data["input_values"]).to_pylist()
        )

        # Pre-compute frame_counts for all possible audio lengths
        frame_counts = model._get_feat_extract_output_lengths(
            torch.arange(0, max_length)
        )

        # Build all queries: (item_idx, t, keypoint)
        # Also compute compression_ratios during this pre-pass
        queries: list[tuple[int, int, int]] = []

        for item_idx in range(num_items):
            audio_length = len(dataset[item_idx]["input_values"])
            fc = frame_counts[:audio_length]
            frame_keypoints = torch.nonzero(fc.diff() > 0).squeeze(1) + 1
            # t ranges from 1 to len(frame_keypoints)-1 (same as original loop)
            for t in range(1, frame_keypoints.shape[0]):
                keypoint = int(frame_keypoints[t].item())
                queries.append((item_idx, t, keypoint))
            # Compression ratio: frame at last valid position / audio_length
            compression_ratios[item_idx] = fc[-1].item() / audio_length

        # Sort queries by keypoint so batches have similar-length audio clips
        queries.sort(key=lambda q: q[2])

        query_ds = _QueryDataset(dataset, queries)

        loader = DataLoader(
            query_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_cuda,
            collate_fn=_collate_pc,
        )

        # Accumulate results keyed by item_idx → list of (frame_pos, hidden_state_row)
        per_item_results: dict[int, list[tuple[int, torch.Tensor]]] = {
            i: [] for i in range(num_items)
        }

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Extracting hidden states (pseudo-causal)"):
                input_values = batch["input_values"].to(device, non_blocking=use_cuda)
                attention_mask = batch["attention_mask"].to(device, non_blocking=use_cuda)

                output = model(
                    output_hidden_states=True,
                    input_values=input_values,
                    attention_mask=attention_mask,
                )

                hidden_layer = output.hidden_states[layer].cpu()  # (B, T, H)
                item_idxs = batch["item_idxs"]
                frame_positions = batch["frame_positions"]

                # Use advanced indexing to extract one row per query into a fresh
                # (B, H) tensor, then let the large (B, T, H) hidden_layer go out
                # of scope immediately so it can be freed.
                batch_len = hidden_layer.shape[0]
                fp_tensor = torch.as_tensor(frame_positions, dtype=torch.long)
                extracted = hidden_layer[torch.arange(batch_len), fp_tensor].clone()  # (B, H)

                for j, (item_idx, frame_pos) in enumerate(
                    zip(item_idxs, frame_positions)
                ):
                    per_item_results[item_idx].append((frame_pos, extracted[j]))

        # Emit results in item order, with each item's frames in frame_pos order
        for idx in range(num_items):
            item_frames = per_item_results[idx]
            # Sort by frame_pos (ascending) to preserve original ordering
            item_frames.sort(key=lambda x: x[0])
            for frame_pos, hidden_row in item_frames:
                flat_idxs.append((idx, frame_pos))
                frame_states_list.append(hidden_row.unsqueeze(0))

    # Concatenate all frame states: (total_frames, hidden_size) → then unsqueeze
    frame_states = torch.cat(frame_states_list, dim=0)
    # For non-pseudo-causal: frame_states is (total_frames, hidden_size) after cat
    # For pseudo-causal: each element is (1, hidden_size), so cat → (total_frames, hidden_size)
    # Add the layer dimension: → (total_frames, 1, hidden_size)
    frame_states = frame_states.unsqueeze(1).contiguous()

    return SpeechHiddenStateDataset(
        model.name_or_path, frame_states, compression_ratios, flat_idxs
    )


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    processor = prepare_processor(config)
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir).with_format("torch")

    model = transformers.Wav2Vec2Model.from_pretrained(config.base_model.model_ref).to(
        config.device
    )

    hidden_state_dataset = extract_hidden_states(
        dataset,
        model,
        processor,
        config.base_model.layer,
        pseudo_causal=config.base_model.pseudo_causal,
    )

    hidden_state_dataset.to_hdf5(config.base_model.hidden_state_path)


if __name__ == "__main__":
    main()

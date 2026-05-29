from functools import partial
from typing import Optional

import datasets
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute
import torch
from torch.utils.data import DataLoader, Dataset
import transformers

from beartype import beartype
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.datasets.speech_equivalence import SpeechHiddenStateDataset
from src.models.transformer import prepare_processor


# ---------------------------------------------------------------------------
# Lightweight Dataset wrappers for DataLoader
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
    """Dataset of (item_idx, t, keypoint) pseudo-causal queries sorted by keypoint."""

    def __init__(self, hf_dataset: datasets.Dataset, queries: list[tuple[int, int, int]]):
        self._ds = hf_dataset
        self._queries = queries

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, pos: int) -> dict:
        item_idx, t, keypoint = self._queries[pos]
        audio = torch.as_tensor(self._ds[item_idx]["input_values"])
        return {
            "audio_clip": audio[:keypoint].clone(),
            "keypoint": keypoint,
            "item_idx": item_idx,
            "frame_pos": t - 1,  # 0-based output frame index
        }


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------


def _collate_npc(batch: list[dict], processor: transformers.Wav2Vec2Processor) -> dict:
    padded = processor.pad(
        [{"input_values": item["input_values"]} for item in batch],
        max_length=None,
        return_tensors="pt",
        return_attention_mask=True,
    )
    padded["orig_idxs"] = [item["orig_idx"] for item in batch]
    return padded


def _collate_pc(batch: list[dict]) -> dict:
    max_len = max(item["keypoint"] for item in batch)
    B = len(batch)
    input_values = torch.zeros(B, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(B, max_len, dtype=torch.int32)
    item_idxs, frame_positions = [], []
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
# Pre-pass: frame layout without GPU work
# ---------------------------------------------------------------------------


def _compute_pre_pass(dataset: datasets.Dataset, model: transformers.Wav2Vec2Model):
    """Compute frame counts, offsets, flat_idxs and compression_ratios from audio lengths."""
    audio_lengths = pa.compute.list_value_length(
        dataset._data["input_values"]
    ).to_pylist()
    num_items = len(audio_lengths)

    frame_counts = [
        int(model._get_feat_extract_output_lengths(torch.tensor(L)).item())
        for L in audio_lengths
    ]
    item_offsets = [0] * num_items
    for i in range(1, num_items):
        item_offsets[i] = item_offsets[i - 1] + frame_counts[i - 1]
    total_frames = item_offsets[-1] + frame_counts[-1] if num_items > 0 else 0

    flat_idxs_array = np.empty((total_frames, 2), dtype=np.int32)
    for i, (offset, count) in enumerate(zip(item_offsets, frame_counts)):
        flat_idxs_array[offset : offset + count, 0] = i
        flat_idxs_array[offset : offset + count, 1] = np.arange(count, dtype=np.int32)

    compression_ratios_array = np.array(
        [fc / al for fc, al in zip(frame_counts, audio_lengths)], dtype=np.float32
    )

    return frame_counts, item_offsets, total_frames, flat_idxs_array, compression_ratios_array


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@beartype
def extract_hidden_states(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    processor: transformers.Wav2Vec2Processor,
    layer: int,
    pseudo_causal: bool = False,
    batch_size: int = 32,
    out_path: Optional[str] = None,
) -> SpeechHiddenStateDataset:
    """Extract wav2vec2 hidden states for every frame in ``dataset``.

    Args:
        dataset: HuggingFace Dataset with ``input_values`` column.
        model: Wav2Vec2Model to extract from.
        processor: Wav2Vec2Processor used for padding.
        layer: Which transformer hidden layer to extract.
        pseudo_causal: If True, mask future audio per output frame.
        batch_size: Items (non-pc) or queries (pc) per GPU batch.
        out_path: If given, stream-write to this HDF5 path (O(batch) peak RAM)
            and return a memory-mapped dataset. If None, accumulate in RAM
            (legacy path, fine for small datasets).
    """
    model.eval()
    if out_path is not None:
        return _extract_streaming(
            dataset, model, processor, layer, pseudo_causal, batch_size, out_path
        )
    return _extract_in_memory(dataset, model, processor, layer, pseudo_causal, batch_size)


# ---------------------------------------------------------------------------
# Streaming path — O(batch) peak memory
# ---------------------------------------------------------------------------


def _extract_streaming(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    processor: transformers.Wav2Vec2Processor,
    layer: int,
    pseudo_causal: bool,
    batch_size: int,
    out_path: str,
) -> SpeechHiddenStateDataset:
    frame_counts, item_offsets, total_frames, flat_idxs_array, compression_ratios_array = (
        _compute_pre_pass(dataset, model)
    )
    hidden_size = model.config.hidden_size
    device = model.device
    use_cuda = device.type == "cuda"

    with h5py.File(out_path, "w") as hf:
        hf.attrs["model_name"] = model.name_or_path
        states_ds = hf.create_dataset(
            "states", shape=(total_frames, 1, hidden_size), dtype=np.float32
        )
        hf.create_dataset("flat_idxs", data=flat_idxs_array, dtype=np.int32)
        hf.create_dataset(
            "compression_ratios", data=compression_ratios_array, dtype=np.float32
        )

        if pseudo_causal:
            _stream_pseudo_causal(
                dataset, model, layer, batch_size, item_offsets, states_ds, device, use_cuda
            )
        else:
            _stream_non_pseudo_causal(
                dataset, model, processor, layer, batch_size,
                item_offsets, frame_counts, states_ds, device, use_cuda,
            )

    return SpeechHiddenStateDataset.from_hdf5(out_path)


def _stream_non_pseudo_causal(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    processor: transformers.Wav2Vec2Processor,
    layer: int,
    batch_size: int,
    item_offsets: list,
    frame_counts: list,
    states_ds: h5py.Dataset,
    device: torch.device,
    use_cuda: bool,
) -> None:
    # Sort items by audio length for length-bucketed batching (minimises padding waste)
    audio_lengths = pa.compute.list_value_length(
        dataset._data["input_values"]
    ).to_pylist()
    sorted_indices = sorted(range(len(audio_lengths)), key=lambda i: audio_lengths[i])

    length_ds = _LengthSortedDataset(dataset, sorted_indices)
    loader = DataLoader(
        length_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
        collate_fn=partial(_collate_npc, processor=processor),
    )

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting hidden states"):
            orig_idxs = batch.pop("orig_idxs")
            output = model(
                output_hidden_states=True,
                input_values=batch["input_values"].to(device, non_blocking=use_cuda),
                attention_mask=batch["attention_mask"].to(device, non_blocking=use_cuda),
            )
            # (B, T, H) → CPU numpy for HDF5 write
            hidden = output.hidden_states[layer].cpu().numpy()
            for j, orig_idx in enumerate(orig_idxs):
                orig_idx = int(orig_idx)
                n = frame_counts[orig_idx]
                off = item_offsets[orig_idx]
                states_ds[off : off + n, 0, :] = hidden[j, :n, :]


def _stream_pseudo_causal(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    layer: int,
    batch_size: int,
    item_offsets: list,
    states_ds: h5py.Dataset,
    device: torch.device,
    use_cuda: bool,
) -> None:
    # Cross-item batching is INCORRECT for pseudo-causal: wav2vec2's convolutional
    # positional encoding (128-frame kernel) is computed over the full padded sequence,
    # so mixing queries from different items changes the context seen by earlier frames
    # and produces different outputs. We must keep within-item batching (same as the
    # in-memory path) and only swap accumulation for direct HDF5 writes.
    audio_lengths = pa.compute.list_value_length(
        dataset._data["input_values"]
    ).to_pylist()
    max_length = max(audio_lengths)
    frame_counts_lookup = model._get_feat_extract_output_lengths(
        torch.arange(0, max_length)
    )
    attention_mask_buf = torch.zeros(batch_size, max_length, dtype=torch.int32).to(device)

    with torch.inference_mode():
        for item_idx in tqdm(range(len(dataset)), desc="Extracting hidden states (pseudo-causal)"):
            item = dataset[item_idx]
            audio = torch.as_tensor(item["input_values"]).unsqueeze(0)
            audio_length = audio.shape[1]

            fc = frame_counts_lookup[:audio_length]
            frame_keypoints = torch.nonzero(fc.diff() > 0).squeeze(1) + 1

            for i in range(1, frame_keypoints.shape[0], batch_size):
                batch_frame_targets = torch.arange(
                    i, min(i + batch_size, frame_keypoints.shape[0])
                )
                batch_keypoints = frame_keypoints[i : i + batch_size]
                batch_length = int(max(batch_keypoints).item())
                real_batch_size = batch_keypoints.shape[0]

                batch_inputs = torch.tile(audio[:, :batch_length].to(device), (real_batch_size, 1))
                for j, kp in enumerate(batch_keypoints):
                    batch_inputs[j, kp:] = 0

                attention_mask_buf.fill_(0)
                for j, kp in enumerate(batch_keypoints):
                    attention_mask_buf[j, :kp] = 1

                output = model(
                    output_hidden_states=True,
                    input_values=batch_inputs,
                    attention_mask=attention_mask_buf[:real_batch_size, :batch_length],
                )

                batch_hidden = output.hidden_states[layer][
                    torch.arange(real_batch_size), batch_frame_targets - 1
                ].cpu().numpy()

                for k in range(real_batch_size):
                    frame_pos = int(batch_frame_targets[k].item()) - 1
                    flat_idx = item_offsets[item_idx] + frame_pos
                    states_ds[flat_idx, 0, :] = batch_hidden[k]


# ---------------------------------------------------------------------------
# In-memory path — legacy, preserved for backward compatibility
# ---------------------------------------------------------------------------


def _extract_in_memory(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    processor: transformers.Wav2Vec2Processor,
    layer: int,
    pseudo_causal: bool,
    batch_size: int,
) -> SpeechHiddenStateDataset:
    flat_idxs: list[tuple[int, int]] = []
    frame_states_list: list[torch.Tensor] = []
    compression_ratios: dict[int, float] = {}

    def collate_batch(batch):
        return processor.pad(
            [{"input_values": v} for v in batch["input_values"]],
            max_length=None,
            return_tensors="pt",
            return_attention_mask=True,
        )

    def extract_representations(batch_items, idxs):
        batch = collate_batch(batch_items)
        with torch.no_grad():
            output = model(
                output_hidden_states=True,
                input_values=batch["input_values"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
            )
        input_lengths = batch["attention_mask"].sum(dim=1)
        frame_lengths = model._get_feat_extract_output_lengths(input_lengths)
        batch_hidden_states = output.hidden_states[layer].cpu()
        batch_compression_ratios = frame_lengths / input_lengths.numpy()
        for idx, num_frames_i, hidden_states_i, compression_i in zip(
            idxs, frame_lengths, batch_hidden_states, batch_compression_ratios
        ):
            flat_idxs.extend([(idx, j) for j in range(num_frames_i)])
            frame_states_list.append(hidden_states_i[:num_frames_i])
            compression_ratios[idx] = compression_i

    def extract_representations_pseudo_causal(item, idx, max_length=None, frame_counts=None):
        assert frame_counts is not None and max_length is not None
        audio = torch.tensor(item["input_values"]).unsqueeze(0)
        audio_length = audio.shape[1]
        frame_counts = frame_counts[:audio_length]
        frame_keypoints = torch.nonzero(frame_counts.diff() > 0).squeeze(1) + 1
        attention_mask = torch.zeros(batch_size, audio_length, dtype=torch.int32).to(model.device)
        for i in range(1, frame_keypoints.shape[0], batch_size):
            batch_frame_targets = torch.arange(i, min(i + batch_size, frame_keypoints.shape[0]))
            batch_keypoints = frame_keypoints[i : i + batch_size]
            batch_length = max(batch_keypoints)
            real_batch_size = batch_keypoints.shape[0]
            batch_inputs = torch.tile(audio[:, :batch_length].to(model.device), (real_batch_size, 1))
            for j, kp in enumerate(batch_keypoints):
                batch_inputs[j, kp:] = 0
            attention_mask.fill_(0)
            for j, kp in enumerate(batch_keypoints):
                attention_mask[j, :kp] = 1
            with torch.no_grad():
                output = model(
                    output_hidden_states=True,
                    input_values=batch_inputs,
                    attention_mask=attention_mask[:real_batch_size, :batch_length],
                )
            batch_hidden_states = output.hidden_states[layer][
                torch.arange(real_batch_size), batch_frame_targets - 1
            ].cpu()
            frame_states_list.append(batch_hidden_states)
            flat_idxs.extend([(idx, j - 1) for j in range(i, i + real_batch_size)])
        compression_ratios[idx] = frame_counts[-1].item() / audio_length

    if pseudo_causal:
        max_length = max(
            pa.compute.list_value_length(dataset._data["input_values"]).to_pylist()
        )
        frame_counts = model._get_feat_extract_output_lengths(torch.arange(0, max_length))
        dataset.map(
            extract_representations_pseudo_causal,
            fn_kwargs={"frame_counts": frame_counts, "max_length": max_length},
            batched=False,
            with_indices=True,
            desc="Extracting hidden states",
        )
    else:
        dataset.map(
            extract_representations,
            batched=True,
            batch_size=batch_size,
            with_indices=True,
            desc="Extracting hidden states",
        )

    frame_states = torch.cat(frame_states_list, dim=0).unsqueeze(1).contiguous()
    return SpeechHiddenStateDataset(model.name_or_path, frame_states, compression_ratios, flat_idxs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    processor = prepare_processor(config)
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir).with_format("torch")
    model = transformers.Wav2Vec2Model.from_pretrained(config.base_model.model_ref).to(
        config.device
    )
    extract_hidden_states(
        dataset,
        model,
        processor,
        config.base_model.layer,
        pseudo_causal=config.base_model.pseudo_causal,
        out_path=config.base_model.hidden_state_path,
    )


if __name__ == "__main__":
    main()

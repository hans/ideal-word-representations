from typing import Optional

import datasets
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute
import torch
import transformers

from beartype import beartype
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.datasets.speech_equivalence import SpeechHiddenStateDataset
from src.models.transformer import prepare_processor


def _compute_pre_pass(dataset: datasets.Dataset, model: transformers.Wav2Vec2Model):
    """
    Compute per-item frame counts, item offsets, total frames, flat_idxs array,
    and compression_ratios array without decoding audio into GPU.

    Returns:
        frame_counts: list[int], length = num_items
        item_offsets: list[int], cumulative start index in flat array per item
        total_frames: int
        flat_idxs_array: np.ndarray shape (total_frames, 2) int32
        compression_ratios_array: np.ndarray shape (num_items,) float32
    """
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

    return (
        frame_counts,
        item_offsets,
        total_frames,
        flat_idxs_array,
        compression_ratios_array,
    )


@beartype
def extract_hidden_states(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    processor: transformers.Wav2Vec2Processor,
    layer: int,
    pseudo_causal: bool = False,
    batch_size: int = 12,
    out_path: Optional[str] = None,
) -> SpeechHiddenStateDataset:
    """Extract hidden states from a wav2vec2 model.

    Args:
        dataset: HuggingFace Dataset with ``input_values`` column.
        model: Wav2Vec2Model to extract from.
        processor: Wav2Vec2Processor used for padding.
        layer: Which transformer hidden layer to extract.
        pseudo_causal: If True, mask future audio frames per output frame.
        batch_size: Number of items / queries per GPU batch.
        out_path: If provided, stream-write directly to this HDF5 file (O(batch)
            memory).  If None, accumulate in RAM and return an in-memory dataset
            (legacy behaviour, for small datasets).

    Returns:
        SpeechHiddenStateDataset (memory-mapped from ``out_path`` when provided).
    """
    model.eval()

    if out_path is not None:
        return _extract_streaming(
            dataset, model, processor, layer, pseudo_causal, batch_size, out_path
        )
    else:
        return _extract_in_memory(
            dataset, model, processor, layer, pseudo_causal, batch_size
        )


# ---------------------------------------------------------------------------
# In-memory path (legacy, unchanged semantics)
# ---------------------------------------------------------------------------


def _extract_in_memory(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    processor: transformers.Wav2Vec2Processor,
    layer: int,
    pseudo_causal: bool,
    batch_size: int,
) -> SpeechHiddenStateDataset:
    flat_idxs = []
    frame_states_list = []
    compression_ratios = {}

    def collate_batch(batch):
        batch = processor.pad(
            [{"input_values": values_i} for values_i in batch["input_values"]],
            max_length=None,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return batch

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

        # batch_size * sequence_length * hidden_size
        batch_hidden_states = output.hidden_states[layer].cpu()

        batch_compression_ratios = frame_lengths / input_lengths.numpy()
        for idx, num_frames_i, hidden_states_i, compression_i in zip(
            idxs, frame_lengths, batch_hidden_states, batch_compression_ratios
        ):
            flat_idxs.extend([(idx, j) for j in range(num_frames_i)])
            frame_states_list.append(hidden_states_i[:num_frames_i])
            compression_ratios[idx] = compression_i

    def extract_representations_pseudo_causal(
        item, idx, max_length=None, frame_counts=None
    ):
        assert frame_counts is not None
        assert max_length is not None

        audio = torch.tensor(item["input_values"]).unsqueeze(0)
        audio_length = audio.shape[1]

        frame_counts = frame_counts[:audio_length]
        frame_keypoints = torch.nonzero(frame_counts.diff() > 0).squeeze(1) + 1

        attention_mask = torch.zeros(batch_size, audio_length, dtype=torch.int32).to(
            model.device
        )

        # NB we start from 1 here.
        # We want the output at frame i to have input sufficient for computing up and through
        # frame i. So we should map output frame i to input keypoint i+1.
        for i in range(1, frame_keypoints.shape[0], batch_size):
            batch_frame_targets = torch.arange(
                i, min(i + batch_size, frame_keypoints.shape[0])
            )
            batch_keypoints = frame_keypoints[i : i + batch_size]
            batch_length = max(batch_keypoints)
            real_batch_size = batch_keypoints.shape[0]

            batch_inputs = audio[:, :batch_length].to(model.device)
            batch_inputs = torch.tile(batch_inputs, (real_batch_size, 1))
            for j, frame_keypoint in enumerate(batch_keypoints):
                batch_inputs[j, frame_keypoint:] = 0

            attention_mask.fill_(0)
            for j, frame_keypoint in enumerate(batch_keypoints):
                attention_mask[j, :frame_keypoint] = 1

            with torch.no_grad():
                output = model(
                    output_hidden_states=True,
                    input_values=batch_inputs,
                    attention_mask=attention_mask[:real_batch_size, :batch_length],
                )

            batch_hidden_states = output.hidden_states[layer][
                torch.arange(real_batch_size), batch_frame_targets - 1
            ].cpu()
            assert len(batch_hidden_states) == real_batch_size
            frame_states_list.append(batch_hidden_states)
            flat_idxs.extend([(idx, j - 1) for j in range(i, i + real_batch_size)])

        compression_ratios[idx] = frame_counts[-1].item() / audio_length

    # Extract and un-pad hidden representations from the model
    if pseudo_causal:
        max_length = max(
            pa.compute.list_value_length(dataset._data["input_values"]).to_pylist()
        )

        # Find the input wav frames at which a new frame is created.
        # pre-calculate and share across the invocations of extract_representations_pseudo_causal
        frame_counts = model._get_feat_extract_output_lengths(
            torch.arange(0, max_length)
        )

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

    frame_states = torch.cat(frame_states_list, dim=0)
    frame_states = frame_states.unsqueeze(1).contiguous()
    # frame_states: total_num_frames * 1 * hidden_size

    return SpeechHiddenStateDataset(
        model.name_or_path, frame_states, compression_ratios, flat_idxs
    )


# ---------------------------------------------------------------------------
# Streaming path (O(batch) peak memory)
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
    """Write hidden states directly to HDF5 during inference."""

    # ------------------------------------------------------------------
    # Pre-pass: compute layout without any GPU work
    # ------------------------------------------------------------------
    (
        frame_counts,
        item_offsets,
        total_frames,
        flat_idxs_array,
        compression_ratios_array,
    ) = _compute_pre_pass(dataset, model)
    hidden_size = model.config.hidden_size

    # ------------------------------------------------------------------
    # Open HDF5 and pre-allocate datasets
    # ------------------------------------------------------------------
    with h5py.File(out_path, "w") as hf:
        hf.attrs["model_name"] = model.name_or_path
        states_ds = hf.create_dataset(
            "states", shape=(total_frames, 1, hidden_size), dtype=np.float32
        )
        hf.create_dataset("flat_idxs", data=flat_idxs_array, dtype=np.int32)
        hf.create_dataset(
            "compression_ratios", data=compression_ratios_array, dtype=np.float32
        )

        # ------------------------------------------------------------------
        # Inference loop: write directly into states_ds
        # ------------------------------------------------------------------
        if pseudo_causal:
            _stream_pseudo_causal(
                dataset, model, layer, batch_size, frame_counts, item_offsets, states_ds
            )
        else:
            _stream_non_pseudo_causal(
                dataset,
                model,
                processor,
                layer,
                batch_size,
                item_offsets,
                frame_counts,
                states_ds,
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
) -> None:
    """Non-pseudo-causal streaming: process batches and write directly to HDF5."""
    num_items = len(item_offsets)

    for batch_start in tqdm(
        range(0, num_items, batch_size), desc="Extracting hidden states"
    ):
        batch_end = min(batch_start + batch_size, num_items)
        batch_items = dataset[batch_start:batch_end]

        # Pad the batch
        batch = processor.pad(
            [{"input_values": v} for v in batch_items["input_values"]],
            max_length=None,
            return_tensors="pt",
            return_attention_mask=True,
        )

        with torch.no_grad():
            output = model(
                output_hidden_states=True,
                input_values=batch["input_values"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
            )

        # shape: (batch, seq_len, hidden_size)
        batch_hidden_states = output.hidden_states[layer].cpu().numpy()

        for j, global_idx in enumerate(range(batch_start, batch_end)):
            n_frames = frame_counts[global_idx]
            offset = item_offsets[global_idx]
            # Write (n_frames, hidden_size) slice, reshaped to (n_frames, 1, hidden_size)
            states_ds[offset : offset + n_frames, 0, :] = batch_hidden_states[
                j, :n_frames, :
            ]


def _stream_pseudo_causal(
    dataset: datasets.Dataset,
    model: transformers.Wav2Vec2Model,
    layer: int,
    batch_size: int,
    frame_counts: list,
    item_offsets: list,
    states_ds: h5py.Dataset,
) -> None:
    """Pseudo-causal streaming: process each item's keypoints and write directly to HDF5."""
    max_length = max(
        pa.compute.list_value_length(dataset._data["input_values"]).to_pylist()
    )
    # pre-calculate frame counts for all input lengths up to max_length
    frame_counts_lookup = model._get_feat_extract_output_lengths(
        torch.arange(0, max_length)
    )

    for item_idx in tqdm(range(len(dataset)), desc="Extracting hidden states"):
        item = dataset[item_idx]
        audio = item["input_values"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        audio = audio.unsqueeze(0)
        audio_length = audio.shape[1]

        fc_item = frame_counts_lookup[:audio_length]
        frame_keypoints = torch.nonzero(fc_item.diff() > 0).squeeze(1) + 1

        attention_mask = torch.zeros(batch_size, audio_length, dtype=torch.int32).to(
            model.device
        )

        # NB we start from 1 here (same as in-memory path).
        for i in range(1, frame_keypoints.shape[0], batch_size):
            batch_frame_targets = torch.arange(
                i, min(i + batch_size, frame_keypoints.shape[0])
            )
            batch_keypoints = frame_keypoints[i : i + batch_size]
            batch_length = max(batch_keypoints)
            real_batch_size = batch_keypoints.shape[0]

            batch_inputs = audio[:, :batch_length].to(model.device)
            batch_inputs = torch.tile(batch_inputs, (real_batch_size, 1))
            for j, frame_keypoint in enumerate(batch_keypoints):
                batch_inputs[j, frame_keypoint:] = 0

            attention_mask.fill_(0)
            for j, frame_keypoint in enumerate(batch_keypoints):
                attention_mask[j, :frame_keypoint] = 1

            with torch.no_grad():
                output = model(
                    output_hidden_states=True,
                    input_values=batch_inputs,
                    attention_mask=attention_mask[:real_batch_size, :batch_length],
                )

            # shape: (real_batch_size, hidden_size)
            batch_hidden_states = (
                output.hidden_states[layer][
                    torch.arange(real_batch_size), batch_frame_targets - 1
                ]
                .cpu()
                .numpy()
            )

            # Write each extracted frame directly to its pre-computed flat index
            for k, frame_pos in enumerate(range(i - 1, i - 1 + real_batch_size)):
                flat_idx = item_offsets[item_idx] + frame_pos
                states_ds[flat_idx, 0, :] = batch_hidden_states[k]


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    processor = prepare_processor(config)
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir).with_format(
        "torch"
    )

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

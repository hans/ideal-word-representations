from pathlib import Path

import datasets
import torch
import transformers

from beartype import beartype
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.datasets.speech_equivalence import SpeechHiddenStateDataset
from src.models.transformer import prepare_processor


@beartype
def extract_hidden_states(dataset: datasets.Dataset,
                          model: transformers.Wav2Vec2Model,
                          processor: transformers.Wav2Vec2Processor,
                          layer: int,
                          batch_size=8) -> SpeechHiddenStateDataset:
    flat_idxs = []
    frame_states_list = []

    def collate_batch(batch):
        batch = processor.pad(
            [{"input_values": values_i} for values_i in batch["input_values"]],
            max_length=None,
            return_tensors="pt",
            return_attention_mask=True)
        return batch

    # Extract and un-pad hidden representations from the model
    def extract_representations(batch_items, idxs):
        batch = collate_batch(batch_items)

        with torch.no_grad():
            output = model(output_hidden_states=True,
                           input_values=batch["input_values"].to(model.device),
                           attention_mask=batch["attention_mask"].to(model.device))

        input_lengths = batch["attention_mask"].sum(dim=1)
        frame_lengths = model._get_feat_extract_output_lengths(input_lengths)

        # batch_size * sequence_length * hidden_size
        batch_hidden_states = output.hidden_states[layer].cpu()

        for idx, num_frames_i, hidden_states_i in zip(idxs, frame_lengths, batch_hidden_states):
            flat_idxs.extend([(idx, j) for j in range(num_frames_i)])
            frame_states_list.append(hidden_states_i[:num_frames_i])

    dataset.map(extract_representations,
                batched=True, batch_size=batch_size,
                with_indices=True,
                desc="Extracting hidden states")
    
    frame_states = torch.cat(frame_states_list, dim=0)
    frame_states = frame_states.unsqueeze(1).contiguous()
    # frame_states: total_num_frames * 1 * hidden_size

    return SpeechHiddenStateDataset(model.name_or_path, frame_states, flat_idxs)


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    processor = prepare_processor(config)
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir)
    
    model = transformers.Wav2Vec2Model.from_pretrained(config.base_model.model_ref).to(config.device)

    hidden_state_dataset = extract_hidden_states(
        dataset, model, processor, config.base_model.layer)
    
    with open(Path(HydraConfig.get().runtime.output_dir) / "hidden_states.pkl", "wb") as f:
        torch.save(hidden_state_dataset, f)


if __name__ == "__main__":
    main()
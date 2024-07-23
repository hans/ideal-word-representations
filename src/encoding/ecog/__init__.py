from dataclasses import dataclass
import logging
from typing import Literal, TypeAlias, Union, cast, Optional

from pathlib import Path

from beartype import beartype
import datasets
import mne
from mne.decoding import ReceptiveField
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.io import loadmat
import torch
from tqdm.auto import tqdm

from src.analysis.state_space import StateSpaceAnalysisSpec
from src.datasets.speech_equivalence import SpeechEquivalenceDataset, SpeechHiddenStateDataset


# processed struct from matlab pipeline
OutFile: TypeAlias = list[dict[str, np.ndarray]]
OutFileWithAnnotations: TypeAlias = list[dict[str, np.ndarray]]
"""
has fields `resp` along with feature ndarrays. first axis is
feature/channel axis; second axis is time axis
"""


L = logging.getLogger(__name__)


@dataclass
class ContrastiveModelSnapshot:
    """
    Snapshot of the training environment and output of a contrastive model used
    for brain encoding.
    """
    # TODO would be nicer to scaffold with Hydra instantiate :)

    feature_spec: DictConfig
    dataset: datasets.Dataset
    hidden_states: SpeechHiddenStateDataset
    equiv_dataset: SpeechEquivalenceDataset
    
    all_state_spaces: dict[str, StateSpaceAnalysisSpec]
    state_space: StateSpaceAnalysisSpec
    embeddings: np.ndarray

    @classmethod
    def from_config(cls, config: DictConfig, feature_spec: DictConfig):
        dataset = cast(datasets.Dataset, datasets.load_from_disk(config.dataset_path))

        with open(feature_spec.equivalence_path, "rb") as f:
            equiv_dataset: SpeechEquivalenceDataset = torch.load(f)
        hidden_states = SpeechHiddenStateDataset.from_hdf5(feature_spec.hidden_state_path)
        with open(feature_spec.state_space_path, "rb") as f:
            all_state_spaces: dict[str, StateSpaceAnalysisSpec] = torch.load(f)
        state_space = all_state_spaces[feature_spec.state_space]
        embeddings = np.load(feature_spec.embeddings_path)

        return cls(
            feature_spec=feature_spec,
            dataset=dataset,
            hidden_states=hidden_states,
            equiv_dataset=equiv_dataset,
            all_state_spaces=all_state_spaces,
            state_space=state_space,
            embeddings=embeddings
        )

    def __post_init__(self):
        assert self.embeddings.shape[0] == self.hidden_states.num_frames


class _AlignmentResult:
    def __init__(self):
        self.name_to_frame_bounds = {}
        self.compression_ratios = {}
        self.name_to_item_idx = {}


def process_item_frame_mapping(item,
                               out_all_names: list[str],
                               frames_by_item: dict[int, tuple[int, int]],
                               out: _AlignmentResult):
    """
    Args:
        item: dict
        out_all_names: list[str]
        frames_by_item: dict[int, tuple[int, int]]
            as in `SpeechHiddenStateDataset.frames_by_item`
        self: AlignedECoGDataset
    """
    # Dataset mapper which computes mapping from item to frame indices, etc.
    # Used within AlignedECoGDataset but defined as function here to allow
    # for pickling.

    name = Path(item["file"]).parent.stem.lower() + "_" + item["id"].lower()
    if name in out_all_names:
        out.name_to_item_idx[name] = item["idx"]

        frame_start, frame_end = frames_by_item[item["idx"]]
        out.name_to_frame_bounds[name] = (frame_start, frame_end)
        out.compression_ratios[name] = (frame_end - frame_start) / len(item["input_values"])


class AlignedECoGDataset:
    """
    Stores alignment between ECoG trials (one trial per sentence) and a
    contrastive model snapshot (training dataset, hidden states, embeddings, etc.).
    """

    audio_sfreq: int = 16000

    def __init__(self, snapshot: ContrastiveModelSnapshot,
                 out: OutFileWithAnnotations):
        self._snapshot = snapshot
        self.dataset = snapshot.dataset
        self.out = out

        out_all_names = [out_i["name"] for out_i in out]
        self.name_to_trial_idx = {name: idx for idx, name in enumerate(out_all_names)}
        self.name_to_item_idx, self.name_to_frame_bounds, self.compression_ratios = {}, {}, {}
        
        alignment_result = _AlignmentResult()
        self.dataset.map(process_item_frame_mapping,
                         fn_kwargs=dict(out_all_names=out_all_names,
                                        frames_by_item=snapshot.hidden_states.frames_by_item,
                                        out=alignment_result))
        self.name_to_item_idx = alignment_result.name_to_item_idx
        self.name_to_frame_bounds = alignment_result.name_to_frame_bounds
        self.compression_ratios = alignment_result.compression_ratios

        self._check_consistency()

    def _check_consistency(self):
        # Make sure that ECoG data and model embeddings are of approximately the same length,
        # modulo sampling differences. Compute length of each sentence in seconds according
        # to two sources:
        comparisons = [(out_i["resp"].shape[1] / out_i["dataf"] - out_i["befaft"].sum(), # remove padding
                        (self.name_to_frame_bounds[out_i['name']][1] - self.name_to_frame_bounds[out_i['name']][0]) / self.compression_ratios[out_i["name"]] / self.audio_sfreq)
                        for out_i in self.out if out_i["name"] in self.name_to_frame_bounds]
        np.testing.assert_allclose(*zip(*comparisons), atol=0.05,
                                err_msg="ECoG data and model embeddings should be of approximately the same length")

    @property
    def total_num_frames(self):
        return self._snapshot.hidden_states.num_frames
    
    @property
    def ecog_sfreq(self):
        return self.out[0]["dataf"]

    @property
    def _mne_info(self) -> mne.Info:
        return mne.create_info(
            ch_names=[f"ch{i}" for i in range(self.out[0]["resp"].shape[0])],
            sfreq=self.out[0]["dataf"],
            ch_types="ecog"
        )
    
    @property
    def _epochs_data(self) -> list[np.ndarray]:
        return [out_i["resp"] for out_i in self.out
                if out_i["resp"].ndim == 2]
    
    @property
    def _mne_raw(self) -> mne.io.RawArray:
        data = np.concatenate(self._epochs_data, axis=1)
        return mne.io.RawArray(data, self._mne_info)

    def iter_trajectories(self, state_space_name: str,
                          trial_idx: Optional[int] = None):
        if state_space_name not in self._snapshot.all_state_spaces:
            raise ValueError(f"Unknown state space {state_space_name}")
        state_space = self._snapshot.all_state_spaces[state_space_name]

        if trial_idx is not None:
            trial_idxs = [trial_idx]
        else:
            trial_idxs = range(len(self.out))

        for trial_idx in trial_idxs:
            name = self.out[trial_idx]["name"]

            # NB this is a half-open range -- so will need to adjust for state-space methods
            item_frame_start, item_frame_end = self.name_to_frame_bounds[name]
            
            trajectories_i = state_space.get_trajectories_in_span(item_frame_start, item_frame_end - 1)

            trial_i = self.out[trial_idx]
            trial_start_padding = trial_i["befaft"][0]  # padding at start of trial, in seconds
            for traj_start, traj_end, traj_label_idx, traj_instance_idx in trajectories_i:
                traj_start_secs = (traj_start - item_frame_start) / self.compression_ratios[name] / self.audio_sfreq
                traj_end_secs = (traj_end - item_frame_start) / self.compression_ratios[name] / self.audio_sfreq

                traj_start_ecog = int((trial_start_padding + traj_start_secs) * trial_i["dataf"]) 
                traj_end_ecog = int((trial_start_padding + traj_end_secs) * trial_i["dataf"])

                traj_start_ecog_nopad = int(traj_start_secs * trial_i["dataf"])
                traj_end_ecog_nopad = int(traj_end_secs * trial_i["dataf"])

                yield {
                    "name": name,
                    "item_idx": self.name_to_item_idx[name],
                    "trial_idx": trial_idx,

                    "state_space": state_space_name,
                    "label_idx": traj_label_idx,
                    "instance_idx": traj_instance_idx,

                    # trajectory span, relative to item onset
                    "span_secs": (traj_start_secs, traj_end_secs),
                    "span_model_frames": (traj_start - item_frame_start, traj_end - item_frame_start),
                    "span_ecog_samples": (traj_start_ecog, traj_end_ecog),
                    "span_ecog_samples_nopad": (traj_start_ecog_nopad, traj_end_ecog_nopad),

                    "item_start_frame": item_frame_start,
                    "item_end_frame": item_frame_end,
                }


def epoch_by_state_space(aligned_dataset: AlignedECoGDataset,
                         state_space_name: str,
                         align_to: Union[Literal["onset"], Literal["offset"]] = "onset",
                         epoch_window: tuple[float, float] = (-0.1, 0.5),
                         baseline_window: Optional[tuple[float, float]] = None,
                         data: Optional[dict[str, np.ndarray]] = None,
                         data_is_padded: bool = True,
                         subset_electrodes: Optional[list[int]] = None,
                         pad_mode: str = "constant",
                         pad_values=0.0,
                         zscore=True,
                         drop_outliers=20.,
                         return_df=False,
                         ) -> Union[tuple[np.ndarray, list[dict]], pd.DataFrame]:
    """
    Args:
        aligned_dataset: AlignedECoGDataset
        state_space_name: str
        epoch_window: tuple[float, float]
        baseline_window: optional tuple[float, float]
        data: dict[str, np.ndarray]
            Map from stimulus name to data (ECoG, residuals, etc.) of shape n_channels * n_samples
        data_is_padded: bool
            Whether data has before/after padding as in raw ECoG response
        subset_electrodes: list[int]
        pad_mode: str
        pad_values
        return_df: bool
    """
    # convert epoch window to ECoG samples
    epoch_window = (int(epoch_window[0] * aligned_dataset.ecog_sfreq),
                    int(epoch_window[1] * aligned_dataset.ecog_sfreq))
    epoch_length = epoch_window[1] - epoch_window[0]

    if baseline_window is not None:
        baseline_window = (int(baseline_window[0] * aligned_dataset.ecog_sfreq),
                           int(baseline_window[1] * aligned_dataset.ecog_sfreq))

    # Check shapes
    if data is not None:
        data_0 = next(iter(data.values()))
        for data_i in data.values():
            assert data_i.ndim == 2
            # All epochs should have same number of channels
            assert data_i.shape[0] == data_0.shape[0]

    epochs, epoch_info = [], []
    state_space = aligned_dataset._snapshot.all_state_spaces[state_space_name]

    trajectories = list(aligned_dataset.iter_trajectories(state_space_name))

    zscore_mean, zscore_std = None, None
    if zscore:
        if data is not None:
            zscore_mean = np.concatenate(list(data.values()), axis=1).mean(axis=1)
            zscore_std = np.concatenate(list(data.values()), axis=1).std(axis=1)
        else:
            seen_trials = set(span["trial_idx"] for span in trajectories)

            all_trial_data = np.concatenate([
                aligned_dataset.out[trial_idx]["resp"]
                for trial_idx in seen_trials
                if aligned_dataset.out[trial_idx]["resp"].ndim == 2
            ], axis=1)
            zscore_mean = all_trial_data.mean(axis=1)
            zscore_std = all_trial_data.std(axis=1)

        if subset_electrodes is not None:
            zscore_mean = zscore_mean[subset_electrodes]
            zscore_std = zscore_std[subset_electrodes]

    for span in tqdm(trajectories):
        trial = aligned_dataset.out[span["trial_idx"]]
        name = trial["name"]

        if data is not None:
            try:
                data_i = data[span["name"]]
            except KeyError:
                continue
        else:
            data_i = trial["resp"]
        
        if data_i.ndim > 2:
            # ignore repeats
            continue
        
        # data_i: n_channels * n_samples

        if data_is_padded:
            start_i, end_i = span["span_ecog_samples"]
        else:
            start_i, end_i = span["span_ecog_samples_nopad"]
        anchor_point = start_i if align_to == "onset" else end_i

        epoch_start_i = max(0, anchor_point + epoch_window[0])
        epoch_end_i = min(data_i.shape[1], anchor_point + epoch_window[1])
        epoch = data_i[:, epoch_start_i:epoch_end_i]

        if subset_electrodes is not None:
            epoch = epoch[subset_electrodes]

        if baseline_window is not None:
            baseline_start_i = max(0, anchor_point + baseline_window[0])
            baseline_end_i = min(data_i.shape[1], anchor_point + baseline_window[1])
            baseline = data_i[:, baseline_start_i:baseline_end_i]
            if subset_electrodes is not None:
                baseline = baseline[subset_electrodes]
            epoch -= baseline.mean(axis=1, keepdims=True)

        if zscore_mean is not None and zscore_std is not None:
            epoch = (epoch - zscore_mean[:, None]) / zscore_std[:, None]

        # Pad if necessary
        epoch_length_i = epoch.shape[1]

        if epoch_length_i < epoch_length:
            pad_left = min(0, anchor_point + epoch_window[0])
            pad_right = max(0, anchor_point + epoch_window[1] - data_i.shape[1])
            pad_width = ((0, 0), (pad_left, pad_right))

            assert pad_left + epoch_length_i + pad_right == epoch_length
            epoch = np.pad(epoch, pad_width, mode=pad_mode, constant_values=pad_values)

        epoch_info.append({
            "epoch_idx": len(epochs),
            "epoch_label": state_space.labels[span["label_idx"]],
            "epoch_duration_samples": epoch_length_i,
            **span
        })
        epochs.append(epoch)

    if drop_outliers:
        if not zscore:
            L.warning("Dropping outliers on raw, non-z-scored values. Are you sure about this?")
        
        mask = np.abs(epochs).max(axis=(1, 2)) < drop_outliers
        epochs = [epoch_i for epoch_i, mask_i in zip(epochs, mask) if mask_i]
        epoch_info = [epoch_info_i for epoch_info_i, mask_i in zip(epoch_info, mask) if mask_i]

        L.info(f"Dropped {len(mask) - mask.sum()} epochs with outliers ({(1 - mask.mean()) * 100 :.2f}%).")

    if return_df:
        if len(epochs) == 0:
            return pd.DataFrame()

        epoch_df = pd.concat([
                pd.DataFrame(ei, index=pd.Index(np.arange(epochs[0].shape[0]) if subset_electrodes is None else subset_electrodes, name="electrode_idx"))
                for ei in epochs
            ], names=["epoch_idx"], keys=range(len(epochs))) \
            .reset_index() \
            .melt(id_vars=["epoch_idx", "electrode_idx"], var_name="epoch_sample", value_name="value")
        epoch_df["epoch_time"] = (epoch_df["epoch_sample"] + epoch_window[0]) / aligned_dataset.ecog_sfreq
        
        # Add epoch metadata
        epoch_df = pd.merge(epoch_df, pd.DataFrame(epoch_info), on="epoch_idx",
                            how="left", validate="many_to_one")
        epoch_df["epoch_duration_secs"] = epoch_df.epoch_duration_samples / aligned_dataset.ecog_sfreq

        return epoch_df

    return np.array(epochs), epoch_info


def get_electrode_df(config, subject: str):
    electrode_path = Path(config.corpus.paths.imaging_path) / subject / "elecs" / "TDT_elecs_all.mat"
    elecs = loadmat(electrode_path, simplify_cells=True)["anatomy"]
    ret = pd.DataFrame(elecs, columns=["electrode_name", "long_name", "type", "roi"]) \
        .set_index("electrode_name", append=True)
    ret.index.set_names("electrode_idx", level=0, inplace=True)
    return ret


class TemporalReceptiveField(ReceptiveField):

    def score(self, X, y):
        # parent class returns one score per output
        scores = super().score(X, y)
        return scores.mean()
    
    def score_multidimensional(self, X, y):
        return super().score(X, y)
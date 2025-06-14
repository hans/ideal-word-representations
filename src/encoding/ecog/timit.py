import logging
from pathlib import Path
import sys
from typing import cast, Optional, Union

import mat73
import numpy as np
from omegaconf import DictConfig, ListConfig
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from tqdm.auto import tqdm

from src.encoding.ecog import TemporalReceptiveField, OutFile, OutFileWithAnnotations, \
    ContrastiveModelSnapshot, AlignedECoGDataset, get_electrode_df


L = logging.getLogger(__name__)


def load_out_file(path) -> dict[str, OutFile]:
    """
    Load preprocessed ECoG data, accounting for different Matlab export
    formats.
    """
    try:
        return loadmat(path, simplify_cells=True)
    except NotImplementedError:
        # Matlab >= 7.3 format. Load using `mat73` and simulate the
        # simplify_cells functionality for the fields we care about
        data = mat73.loadmat(path)

        # Simplify cell representation
        target_cells = ["name", "resp", "dataf", "befaft"]
        ret = []
        for i in range(len(data["out"]["resp"])):
            ret.append({
                cell: data["out"][cell][i] for cell in target_cells
            })

        return {"out": ret}


def prepare_out_file_fomo(config: DictConfig, data_spec: DictConfig) -> OutFileWithAnnotations:
    # prepare list of grid electrode indices
    electrode_df = get_electrode_df(config, data_spec.subject)
    keep_elecs = electrode_df[electrode_df.type == "grid"]
    
    keep_elec_idxs = keep_elecs.index.get_level_values("electrode_idx").tolist()
    if keep_elec_idxs != list(range(keep_elecs.shape[0])):
        # grid electrodes should start at 0 and be contiguous. we can definitely incorporate
        # other grids but it would require more careful handling on the results end
        drop_point = next((i for i, x in enumerate(keep_elec_idxs) if x != i), None)
        if drop_point == 0:
            L.error("Grid electrodes do not start at 0; skipping")
            return []
        elif drop_point is not None:
            drop_point -= 1
            L.warning(f"Non-contiguous grid electrodes detected; dropping after {keep_elec_idxs[drop_point]}")
            keep_elec_idxs = keep_elec_idxs[:drop_point]

    if len(keep_elec_idxs) == 0:
        L.warning("No grid electrodes found; skipping")
        return []

    from data_utils import data_loader
    from fomo_config import FoMoConfig
    patient = data_loader.nfm_patient(FoMoConfig.load(), data_spec.subject)
    ecog_data = patient.load_task_ecog(
        "TIMIT", "trial", keep_bands="hga",
        include_task_levels=["trial"])

    if ecog_data is None:
        return []
    task_event_dfs, block_ecog_activity = ecog_data

    # sanity checks
    assert len(block_ecog_activity) == len(task_event_dfs)
    for block_ecog, task_event_df in zip(block_ecog_activity, task_event_dfs):
        assert len(block_ecog) == len(task_event_df)

    if len(block_ecog_activity) == 0:
        # no trials. stop
        return []
    
    # if some blocks have fewer numbers of electrodes, subset the electrodes in all blocks to match this
    # assume that alignment of electrodes between remaining channels is correct
    block_channel_counts = [block_ecog_i[0].shape[1] for block_ecog_i in block_ecog_activity
                            if len(block_ecog_i) > 0]
    if len(set(block_channel_counts)) > 1:
        L.warning(f"Blocks have different numbers of electrodes: {block_channel_counts}. Subsetting to minimum of {min(block_channel_counts)}")
        min_channels = min(block_channel_counts)
        block_ecog_activity = [[trial_ecog[:, :min_channels] for trial_ecog in block_ecog_i]
                               for block_ecog_i in block_ecog_activity]

        if max(keep_elec_idxs) >= min_channels:
            L.warning(f"Keeping electrodes {keep_elec_idxs} but only {min_channels} are "
                       "available in all blocks; subsetting")
            keep_elec_idxs = [x for x in keep_elec_idxs if x < min_channels]

    # compute average HGA over 8 bands; subset grid electrodes
    all_trials = [[trial[:, keep_elec_idxs].mean(axis=0) for trial in block_ecog]
                  for block_ecog in block_ecog_activity]
    
    # prepare sentence annotations
    strffeat = loadmat(f"{config.corpus.paths.info_path}/out_sentence_details_timit_all_loudness.mat",
                       simplify_cells=True)['sentdet']
    strffeat = {x["name"]: x for x in strffeat}

    def get_trial_annotations(trial_name):
        trial = strffeat[trial_name]
        ret = {}

        # adding f0 to out struct, should be added to loudness all
        # before adddetails_to_out
        soundir='neural_preprocessing/';
        pitch_file = Path(config.corpus.paths.info_path).parent / soundir / "TimitSounds" / f"{trial_name}_pitch_int.txt"

        # Drop first and last rows
        pitch_df = pd.read_csv(pitch_file, sep='\t', header=None, na_values=["--undefined--"]) \
            .iloc[1:-1][3].astype(float).fillna(0.)
        # Pad with zeros by referencing stress data
        before_pad = int(np.floor((len(trial["stress"]) - len(pitch_df)) / 2))
        after_pad = int(np.ceil((len(trial["stress"]) - len(pitch_df)) / 2))
        trial["f0"] = np.pad(pitch_df, (before_pad, after_pad))[None, :]

        ####

        ret["onset"] = trial["onsOff"][0]
        ret["offset"] = trial["onsOff"][1]

        # phnFeatConsOnset index | phnfeatonset index | description
        # 0 | 0  | dorsal (sh, k, kcl, g, gcl, eng, ng, ch, jh)
        # 1 | 1  | coronal (s, z, t, tcl, d, dcl, n, ch, jh)
        # 2 | 2  | labial (f, v, p, pcl, b, bcl, m, em)
        # 3 | 7  | plosive (p, pcl, t, tcl, k, kcl, b, bcl, d, dcl, g, gcl)
        # 4 | 8  | fricative (f, v, th, dh, s, sh, z, zh, hh, hv, ch, jh)
        # 5 | 10 | nasal (m, em, n, en, ng, eng, nx)
        ret["phnfeatConsOnset"] = trial["phnfeatonset"][[0, 1, 2, 7, 8, 10]]
        
        ret["maxDtL"] = trial["loudnessall"][5]

        # auditory and spectral features
        ret["aud"] = trial["aud"][:80]
        ret["F0"] = trial["f0"][0]
        ret["formantMedOnset"] = trial["frmMedOns"][:4]

        # metadata
        metadata = {
            "befaft": trial["befaft"],
            "dataf": trial["dataf"],
            "soundf": trial["soundf"],
        }

        ret = {key: value[None, :] if value.ndim == 1 else value
            for key, value in ret.items()}
        return ret, metadata

    all_trial_dicts = []
    for block_event_df, block_ecog in zip(tqdm(task_event_dfs, unit="block", desc="Preparing ECoG and stimulus annotations"),
                                          all_trials):
        trial_df = block_event_df[(block_event_df.level == "trial")]
        for trial_idx_in_block, (trial_idx, trial_row) in enumerate(trial_df.iterrows()):
            trial_name = trial_row.stimulus_name
            trial_ecog = block_ecog[trial_idx_in_block]

            annots, metadata = get_trial_annotations(trial_name)

            pad_left, pad_right = metadata["befaft"] if len(metadata["befaft"]) == 2 else (0.5, 0.5)
            sfreq = metadata["dataf"] or 100

            all_trial_dicts.append({
                "name": trial_name,
                "resp": trial_ecog,
                "befaft": np.array([pad_left, pad_right]),
                "dataf": sfreq,
                "soundf": metadata["soundf"],
                **annots,
            })

    # sanity checks
    # same features for all trials
    first_trial_keys = set(all_trial_dicts[0].keys())
    for trial in all_trial_dicts[1:]:
        assert set(trial.keys()) == first_trial_keys
    # same feature dimensionality for all trials (first axis)
    for key in first_trial_keys:
        if not isinstance(all_trial_dicts[0][key], np.ndarray):
            continue
        assert all(trial[key].shape[0] == all_trial_dicts[0][key].shape[0] for trial in all_trial_dicts)

    return all_trial_dicts


def prepare_out_file(config: DictConfig, data_spec: DictConfig) -> OutFileWithAnnotations:
    if getattr(data_spec, "data_source", None) == "fomo":
        return prepare_out_file_fomo(config, data_spec)

    data_dir = Path(config.corpus.paths.data_path) / data_spec.subject / config.corpus.name / "block_z"

    if data_spec.block is None:
        full_glob = f"{data_spec.subject}_*_{config.corpus.paths.out_file_glob}"
    else:
        full_glob = f"{data_spec.subject}_{data_spec.block}_{config.corpus.paths.out_file_glob}"
    outfile = list(data_dir.glob(full_glob))
    assert len(outfile) == 1

    cout = load_out_file(outfile[0])
    out = cout["out"]

    # add sentence details to out
    out = add_details_to_out(out, config.corpus.sentdetV,
                             config.corpus.paths.info_path,
                             config.corpus.name,
                             data_spec.subject)

    return out


def load_and_align_model_embeddings(config, out: OutFileWithAnnotations):
    """
    Load model embeddings of TIMIT stimuli and align them with the
    corresponding ECoG responses / baseline features.
    """

    if len(config.feature_sets.model_features) != 1:
        raise NotImplementedError("Only one model feature set is supported")
    feature_spec = next(iter(config.feature_sets.model_features.values()))
    model = ContrastiveModelSnapshot.from_config(config, feature_spec)

    aligned_dataset = AlignedECoGDataset(model, out)

    # normalize across whole embedding set (not just those used in the encoding evaluation)
    embedding_normalization = getattr(feature_spec, "normalization", None)
    if embedding_normalization == "zscore":
        model.embeddings = (model.embeddings - model.embeddings.mean(axis=0)) / model.embeddings.std(axis=0)
    elif embedding_normalization == "minmax":
        model.embeddings = (model.embeddings - model.embeddings.min(axis=0)) / (model.embeddings.max(axis=0) - model.embeddings.min(axis=0))
    elif embedding_normalization == "l2norm":
        model.embeddings = model.embeddings / np.linalg.norm(model.embeddings, axis=1, keepdims=True)
    elif embedding_normalization is not None:
        raise ValueError(f"Unknown normalization {embedding_normalization}")

    # Scatter model embeddings into a time series aligned with ECoG data.
    n_model_dims = model.embeddings.shape[1]
    embedding_scatter_samples, embedding_scatter_data = [], []

    for traj in aligned_dataset.iter_trajectories(feature_spec.state_space):
        unit_start = traj.item_start_frame + traj.span_model_frames[0]
        unit_end = traj.item_start_frame + traj.span_model_frames[1]

        # sanity check
        if unit_start > model.embeddings.shape[0] or unit_end > model.embeddings.shape[0]:
            raise ValueError("Unit span out of bounds for model embeddings")

        # Prepare embedding of the given unit
        unit_embedding = None
        if feature_spec.featurization == "last_frame":
            unit_embedding = model.embeddings[unit_end]
        elif feature_spec.featurization == "mean":
            # this is an inclusive span; make sure the slice gets at least one sample
            unit_end = max(unit_end, unit_start + 1)
            unit_embedding = model.embeddings[unit_start:unit_end].mean(axis=0)
        elif isinstance(feature_spec.featurization, (tuple, list, ListConfig)):
            if feature_spec.featurization[0] == "mean_last_k":
                k = int(feature_spec.featurization[1])
                mean_start = max(unit_end - k, unit_start)
                # this is an inclusive span, make sure the slice gets at least one sample
                mean_end = max(unit_end, mean_start + 1)
                unit_embedding = model.embeddings[mean_start:mean_end].mean(axis=0)
            else:
                raise ValueError(f"Unknown featurization {feature_spec.featurization}")
        else:
            raise ValueError(f"Unknown featurization {feature_spec.featurization}")
        
        # Scatter an impulse predictor at the sample aligned to the onset of the unit
        embedding_scatter_samples.append((traj.trial_idx, traj.span_ecog_samples[0]))
        embedding_scatter_data.append(unit_embedding)

    embedding_scatter_samples = np.array(embedding_scatter_samples)
    embedding_scatter_data = np.array(embedding_scatter_data)

    if np.isnan(embedding_scatter_data).any():
        raise ValueError("NaNs detected in model embeddings")

    if feature_spec.permute == "permute_units":
        # Permute mapping between token units and their embeddings
        perm = np.random.permutation(embedding_scatter_data.shape[0])
        embedding_scatter_samples = embedding_scatter_samples[perm]
    elif feature_spec.permute == "shift":
        assert "tmax" in config.model and "tmin" in config.model, \
            "Shift permutation requires 'tmin' and 'tmax' to be set in the model configuration"

        # Apply a fixed random jitter to the alignment of the model embeddings,
        # beyond the temporal window of the TRF model
        window_size = np.round((config.model.tmax - config.model.tmin) * config.model.sfreq).astype(int)
        jitter = int(np.random.uniform(window_size, 2 * window_size))
        jitter = (-1 if np.random.rand() < 0.5 else 1) * jitter
        L.info(f"Applying jitter of {jitter} samples to model embeddings")
        embedding_scatter_samples[:, 1] += jitter
    elif feature_spec.permute is None:
        pass
    else:
        raise ValueError(f"Unknown permutation {feature_spec.permute}")

    # Now scatter the model embeddings into the annotated data
    out_of_bounds_units = 0
    for out_i in out:
        out_i["model_embedding"] = np.zeros((n_model_dims, out_i["resp"].shape[1]))
    for (trial_idx, sample), embedding in zip(embedding_scatter_samples, embedding_scatter_data):
        # NB this sample may be out-of-bounds if we shift-permuted and jittered too far
        if sample < 0 or sample >= out[trial_idx]["model_embedding"].shape[1]:
            L.warning(f"Sample {sample} out of bounds for trial {trial_idx}; skipping")
            out_of_bounds_units += 1
            continue
        out[trial_idx]["model_embedding"][:, sample] = embedding

    L.info(f"Scattered model embeddings for {len(embedding_scatter_samples)} {feature_spec.state_space} units")
    if out_of_bounds_units:
        L.warning(f"Skipped {out_of_bounds_units} units ({out_of_bounds_units / len(embedding_scatter_samples) * 100 : .2f}%) due to out-of-bounds samples")

    return out


def prepare_xy(config: DictConfig, data_spec: DictConfig,
               center_features=True, scale_features=True) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int]], np.ndarray]:
    out = prepare_out_file(config, data_spec)
    feature_sets = config.feature_sets.baseline_features[:]

    center_features = [center_features] * len(feature_sets)
    scale_features = [scale_features] * len(feature_sets)

    # add model embeddings to out
    if getattr(config.feature_sets, "model_features", None):
        model_config = next(iter(config.feature_sets.model_features.values()))
        # model config may be present but not complete. only run if we have a full
        # model spec
        if getattr(model_config, "model", None):
            out = load_and_align_model_embeddings(config, out)
            feature_sets.append("model_embedding")

            center_features.append(False)
            scale_features.append(False)

    fit_response_dimensions = None
    if "fit_channels" in data_spec:
        fit_response_dimensions = range(data_spec.fit_channels.start, data_spec.fit_channels.end)

    X, Y, feature_names, feature_shapes, trial_onsets = prepare_strf_xy(
        out, feature_sets, data_spec.subject,
        fit_response_dimensions=fit_response_dimensions,
        center_features=np.array(center_features),
        scale_features=np.array(scale_features))
    
    return X, Y, feature_names, feature_shapes, trial_onsets


def add_details_to_out(out: OutFile, sentdetV, infopath, corpus, cs) -> OutFileWithAnnotations:
    frmVows = set();  allFrm = []; vowId=None; subjFrm=None;
    if corpus != "timit":
        raise NotImplementedError("Only timit is supported")

    # load phoneme features and loudness (from liberty)
    if sentdetV == "v4":
        strffeat = loadmat(Path(infopath, 'out_sentence_details_timit_all_loudness.mat'),
                           simplify_cells=True)['sentdet']
    else:
        raise NotImplementedError("Only v4 is supported")
    
    strffeat_names = [strffeat[i]["name"] for i in range(len(strffeat))]
    addfields = set(strffeat[0].keys()) - set(out[0].keys())

    # allFrm = [strffeat.frmMedVal];

    for i, trial in enumerate(out):
        name_i = trial["name"]
        cl = strffeat_names.index(name_i)
        for field in addfields:
            trial[field] = strffeat[cl][field]
        
        # adding f0 to out struct, should be added to loudness all
        # before adddetails_to_out
        soundir='neural_preprocessing/';
        pitch_file = Path(infopath).parent / soundir / "TimitSounds" / f"{name_i}_pitch_int.txt"

        # Drop first and last rows
        pitch_df = pd.read_csv(pitch_file, sep='\t', header=None, na_values=["--undefined--"]) \
            .iloc[1:-1][3].astype(float).fillna(0.)
        # Pad with zeros by referencing stress data
        before_pad = int(np.floor((len(trial["stress"]) - len(pitch_df)) / 2))
        after_pad = int(np.ceil((len(trial["stress"]) - len(pitch_df)) / 2))
        trial["f0"] = np.pad(pitch_df, (before_pad, after_pad))[None, :]

        # aggregate all scaleflagheard formant values
        if subjFrm is None:
            subjFrm = trial["frmMedVal"][:2, :]
        else:
            subjFrm = np.hstack((subjFrm, trial["frmMedVal"][:2, :]))

        # aggregate all vowel types for current corpus
        frmVows |= set(trial["vowel"])

        # aggregate all vowel ids
        if vowId is None:
            vowId = trial["vowelId"]
        else:
            vowId = np.hstack((vowId, trial["vowelId"]))

        trial["onset"] = trial["onsOff"][0]
        trial["offset"] = trial["onsOff"][1]

        trial["phnfeatConsOnset"] = trial["phnfeatonset"][[0, 1, 2, 7, 8, 10]]
        
        trial["maxDtL"] = trial["loudnessall"][5]

        # auditory and spectral features
        trial["aud"] = trial["aud"][:80]
        trial["F0"] = trial["f0"][0]
        trial["formantMedOnset"] = trial["frmMedOns"][:4]

        # add derivatives of existing fields - {'frmMedOnsbin'}
        # TODO you don't use this right?
        # out = makeBinnedFormantFeature(i, out, 'v1')


    # TODO pitch/formant logic is duplicated below -- why? maybe I got the block structure wrong in the code
    # if corpus in ["timit", "dimex"]:
    #     # all corpus formants (together for f2dmap_v3)
    #     # TODO why the reference to dimex?
    #     allcorpFrm = np.hstack((allcorpFrm, trial["frmMedVal"]))
        
        # Dvow=loadDD(corpus, allFrm);
        
        # % all added features
        # for i = 1:length(out)
                        
        #     % using midpoint (using all sentences)
        #     % out = makeFormantMapFeature(i, out, allFrm, vowId, corpus, 'allsent');
        #     % using midpoint (using subject specific sentences)
        #     % out = makeFormantMapFeature(i, out, subjFrm, vowId, corpus, 'subjsent', cs);
            
        #     %%  optional if model is used
        # %     make all of the other strf features
        #     % relative formant median onset binned 
        # %     out = makeBinnedFormantFeature(i, out, 'v6');
        # % %   relative continuous formant binned
        # %     out = makeBinnedFormantFeature(i, out, 'v8');    
        # % %   log scaled binned continuous formant 
        # %     out = makeBinnedFormantFeature(i, out, 'v9');
        # %     out = makeBinnedFormantFeature(i, out, 'v11');
        # %     out = makeBinnedFormantFeature(i, out, 'v12');
        # %     out = makeBinnedFormantFeature(i, out, 'v13');
        # %     out = makeBinnedFormantFeature(i, out, 'v14');
        # %     out = makeBinnedFormantFeature(i, out, 'v15');
        # %     out = makeBinnedFormantFeature(i, out, 'v16');
        # %     out = makeVowelTypeFeature(i, out, frmVows);
        # % 
        # %     without midpoint (using same map for both corpus, 10 by 10)
        # %     out = makeFormantMapFeature_v3(i, out, vowId, allcorpFrm);
            
        # %     without midpoint (using all sentences)
        # %     out = makeFormantMapFeature_v2(i, out, allFrm(1:2, :), vowId, corpus, 'allsent');
            
        # %     without midpoint (using all sentences, but subject specific sentences)
        # %     out = makeFormantMapFeature_v2(i, out, subjFrm, vowId, corpus, 'subjsent', cs);
        # %     
        # %     formant interaction effects
        # %     out = makeFormantValEffect(i, out, 'v1', allFrm(1:2, :), vowId);
        # %     out = makeFormantValEffect(i, out, 'v2', allFrm(1:2, :), vowId);
            
        #     %% non-optional
        #     % add pitch to out struct
        #     out = makeF0(i, out, Dvow); 
            
        #     % normalized formant values
        #     out = makeNormFormantVals(i, out, Dvow);
            
        #     % add median offset to the out struct
        #     out = makeMedianFormantOffset(i, out);
            
        #     % add vowelType
        #     out = makeVowelCategory(i, out);    
    
    # # add relative binned pitch (not doing this per sentence, because normalize
    # # over entire sentence)
    # numBins = 10;
    # out = makeBinnedPitchFeat(out, numBins);
    
    # # add relative loudness features
    # numBins = 10;
    # out = makeRelLoudness(out, numBins, 'maxDtL');
    
    # # add surprisal features
    # numBins = 8;
    # out = makeSurprisal(out, numBins, corpus);
    # if strcmp(corpus, 'timit')
    #     out = makeSurprisal(out, numBins, [corpus '_cross']);
    # end
    
    # plot formant variance map
    # out = strf_plotFormantVariance(out);

    return out


def prepare_strf_xy(out: OutFileWithAnnotations,
                    feature_sets: list[str], subject_id: str,
                    fit_response_dimensions: Optional[list[int]] = None,
                    center_features: Union[bool, np.ndarray] = True,
                    scale_features: Union[bool, np.ndarray] = True,
                    response_field="resp") -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, ...], np.ndarray]:
    """
    Prepare TRF design matrix.

    Returns:
    - X: np.ndarray, shape (n_samples, n_features)
    - Y: np.ndarray, shape (n_samples, n_outputs)
    - feature_names: list[str], length n_features
    - feature_shapes: tuple[int, ...], length n_features
    - trial_onsets: np.ndarray, shape (n_trials, 2), trial index and sample at which this trial has onset
    """
    response_field = response_field or 'resp'
    
    # estimation parameters
    if isinstance(center_features, bool):
        center_features = center_features * np.ones(len(feature_sets))
    if isinstance(scale_features, bool):
        scale_features = scale_features * np.ones(len(feature_sets))

    # paper strfs
    dataf = out[0]["dataf"];
    fs = out[0]["soundf"]
    # assert dataf == fs, f"Neural and sound sampling rate must match ({dataf} Hz != {fs} Hz)"

    X, Y, trial_names, trial_idxs = [], [], [], []
    feature_names, feature_shapes =  None, None
    for i, trial in enumerate(out):
        # TODO what to do about repeats?
        Yi = trial[response_field]
        if Yi.ndim == 3:
            # TODO this sentence has repeats
            continue
    
        Xi = []
        feature_names_i, feature_shape_i, apparent_num_feature_samples = [], [], None
        for feature_lookup in feature_sets:
            feature = trial[feature_lookup]

            # Check that feature num samples matches what we have so far
            if apparent_num_feature_samples is not None:
                if feature.shape[-1] != apparent_num_feature_samples:
                    if feature.shape[-1] > apparent_num_feature_samples:
                        if i == 0:
                            L.warning(f"Feature {feature_lookup} has {feature.shape[-1]} samples, expected {apparent_num_feature_samples}. Trimming but you should check this!")
                        feature = feature[:, :apparent_num_feature_samples]
                    else:
                        raise ValueError(f"Feature {feature_lookup} has {feature.shape[-1]} samples, expected {apparent_num_feature_samples}")
            else:
                apparent_num_feature_samples = feature.shape[-1]

            if feature.ndim == 2:
                for j, feature_row in enumerate(feature):
                    Xi.append(feature_row)

                    feature_names_i.append(f"{feature_lookup}_{j}")
                feature_shape_i.append(feature.shape[0])
            else:
                if feature.dtype.kind != "f":
                    if feature_names is None:
                        # This is the first trial we're processing; issue a warning
                        # and continue
                        L.warning(f"Coercing non-float feature {feature_lookup} with dtype {feature.dtype} to float")

                    feature = feature.astype(float)
                Xi.append(feature)
                feature_names_i.append(feature_lookup)
                feature_shape_i.append(1)

        if apparent_num_feature_samples != Yi.shape[-1]:
            if apparent_num_feature_samples > Yi.shape[-1]:
                L.warning(f"Feature samples ({apparent_num_feature_samples}) do not match response samples ({Yi.shape[-1]}). Trimming features")
                Xi = [xi[:Yi.shape[-1]] for xi in Xi]
            else:
                L.warning(f"Feature samples ({apparent_num_feature_samples}) do not match response samples ({Yi.shape[-1]}). Trimming response")
                Yi = Yi[:, :apparent_num_feature_samples]

        # print(list(zip([xi.shape for xi in Xi], feature_names_i)))
        Xi = np.stack(Xi).astype(float).T

        # Remove padding
        padding = tuple(trial.get("befaft", (0.5, 0.5)))
        start_sample = int(padding[0] * dataf)
        end_sample = Xi.shape[0] - int(padding[1] * dataf)

        Xi = Xi[start_sample:end_sample]
        Yi = Yi[:, start_sample:end_sample]

        X.append(Xi)
        Y.append(Yi)
        trial_names.append(trial["name"])
        trial_idxs.append(i)

        if feature_names is None:
            feature_names = feature_names_i
            feature_shapes = feature_shape_i
        else:
            assert feature_names == feature_names_i
            assert feature_shapes == feature_shape_i

    # TODO sound onset -- strf_makeXtimeLag:79
    # TODO boundaries
    # record onset of each trial in concatenated array
    trial_onsets = np.concatenate([[0], np.cumsum([Xi.shape[0] for Xi in X])[:-1]])
    trial_onsets = np.stack([np.array(trial_idxs), trial_onsets], axis=1)
    X = np.concatenate(X)
    Y = np.concatenate(Y, axis=1).T
    
    # Center and scale features
    center_features = np.concatenate([np.ones(shape) * center for shape, center in zip(feature_shapes, center_features)])
    scale_features = np.concatenate([np.ones(shape) * scale for shape, scale in zip(feature_shapes, scale_features)]).astype(bool)
    X -= X.mean(axis=0) * center_features

    scale_denoms = X.max(axis=0) - X.min(axis=0)
    scale_denoms[~scale_features] = 1
    X /= scale_denoms

    # Subset instructed electrodes
    if fit_response_dimensions is not None:
        Y = Y[:, fit_response_dimensions]

    Y_range = Y.max(axis=0) - Y.min(axis=0)
    if (Y_range == 0).any():
        null_response_dimensions = np.where(Y_range == 0)[0]
        L.warning(f"Response dimensions {null_response_dimensions} have zero variance. Leaving them in; this will produce normalization errors below")

    # Normalize response. NB this will log a warning for zero-variance channels
    Y -= Y.mean(axis=0)
    Y /= (Y.max(axis=0) - Y.min(axis=0))

    nan_response_dimensions = np.where(np.isnan(Y).any(axis=0))[0]
    if len(nan_response_dimensions):
        L.warning(f"Response dimensions {nan_response_dimensions} contain NaNs. Keeping them and filling with zero to avoid crashes")
        Y[:, nan_response_dimensions] = 0

    return X, Y, feature_names, feature_shapes, trial_onsets


def concat_xy(all_xy: list[tuple[np.ndarray, np.ndarray, list[str], tuple[int, ...], np.ndarray]]):
    X, Y, feature_names, feature_shapes, trial_onsets = [], [], None, None, []
    for Xi, Yi, feature_names_i, feature_shapes_i, trial_onsets_i in all_xy:
        X.append(Xi)
        Y.append(Yi)
        trial_onsets.append(trial_onsets_i)

        if feature_names is None:
            feature_names = feature_names_i
            feature_shapes = feature_shapes_i
        else:
            assert feature_names == feature_names_i
            assert feature_shapes == feature_shapes_i

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    trial_onsets = np.concatenate(trial_onsets)

    # TODO handle boundaries

    return X, Y, feature_names, feature_shapes, trial_onsets


def strf_nested_cv(X, Y, feature_names, feature_shapes, sfreq,
                   trf_kwargs=None,
                   hparams=None,
                   cv_outer=None, cv_inner=None,
                   ) -> tuple[Optional[TemporalReceptiveField],
                              list[np.ndarray],
                              list[np.ndarray],
                              list[np.ndarray],
                              list[dict]]:
    """
    Returns:
    - best_model: TemporalReceptiveField
    - preds: list[np.ndarray], `n_folds` many arrays of shape (n_samples, n_outputs)
    - scores: list[np.ndarray], `n_folds` many arrays of shape (n_outputs,)
    - coefs: list[np.ndarray], `n_folds` many arrays of shape ...
    - best_hparams: list[dict], `n_folds` many dictionaries
    """
    default_trf_kwargs = {
        "tmin": 0, "tmax": 0.6,
        "fit_intercept": True,
        "n_jobs": 1,
        "sfreq": sfreq,
    }
    trf_kwargs = {**default_trf_kwargs, **(trf_kwargs or {})}
    regression = TemporalReceptiveField(
        feature_names=feature_names,
        **trf_kwargs)

    if cv_outer is None or isinstance(cv_outer, int):
        cv_outer = KFold(n_splits=cv_outer or 5, shuffle=False)
    if cv_inner is None or isinstance(cv_inner, int):
        cv_inner = KFold(n_splits=cv_inner or 5, shuffle=False)

    if hparams is None:
        hparams = {'estimator': np.logspace(0, 7, 5)}

    best_estimator = None
    preds, scores, coefs, best_hparams = [], [], [], []
    for i, (train, test) in enumerate(tqdm(cv_outer.split(X, Y), total=cv_outer.get_n_splits())):    
        model = GridSearchCV(regression, hparams,
                             cv=cv_inner,
                             verbose=1,
                             error_score=np.nan,
                             n_jobs=2)

        try:
            model.fit(X[train], Y[train])
        except ValueError as e:
            if "fits failed" in e.args[0]:
                L.warning(f"Failed to fit model on fold {i}; skipping")
                continue
            else:
                raise

        # Check if we picked an alpha at the edge of the hparam space;
        # if so, log a warning
        best_alpha = model.best_params_["estimator"]
        if best_alpha == hparams["estimator"].min():
            L.warning(f"Best alpha {best_alpha} is at the lower edge of the search space")
        elif best_alpha == hparams["estimator"].max():
            L.warning(f"Best alpha {best_alpha} is at the upper edge of the search space")

        best_estimator: TemporalReceptiveField = model.best_estimator_
        preds.append(best_estimator.predict(X[test]))
        scores.append(best_estimator.score_multidimensional(X[test], Y[test]))
        coefs.append(best_estimator.coef_)
        best_hparams.append(model.best_params_)

    return best_estimator, preds, scores, coefs, best_hparams


def trf_grid_to_df(base_model: TemporalReceptiveField, coefs, output_names=None):
    trf_df = []
    for fold, fold_coefs in enumerate(coefs):
        for input_dim, name in enumerate(base_model.feature_names):
            for output_dim in range(fold_coefs.shape[0]):
                local_coefs = fold_coefs[output_dim, input_dim]
                for delay, coef in enumerate(local_coefs):
                    trf_df.append({
                        "fold": fold,
                        "input_dim": input_dim,
                        "feature": name,
                        "output_dim": output_dim,
                        "lag": delay,
                        "coef": coef,
                    })

    trf_df = pd.DataFrame(trf_df)
    if output_names is not None:
        trf_df["output_name"] = trf_df.output_dim.map(dict(enumerate(output_names)))

    trf_df["time"] = trf_df.lag / base_model.sfreq - base_model.tmin

    return trf_df
# %%

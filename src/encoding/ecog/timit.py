from pathlib import Path
from typing import Optional

from mne.decoding import ReceptiveField, TimeDelayingRidge
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from tqdm.auto import tqdm

from src.encoding.ecog import TemporalReceptiveField


def add_details_to_out(out, sentdetV, infopath, corpus, cs):
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
        print(name_i)
        cl = strffeat_names.index(name_i)
        for field in addfields:
            trial[field] = strffeat[cl][field]
        
        # adding f0 to out struct, should be added to loudness all
        # before adddetails_to_out
        soundir='neural_preprocessing/';
        pitch_file = Path(infopath).parent / soundir / "TimitSounds" / f"{name_i}_pitch_int.txt"

        # Drop first and last rows
        pitch_df = pd.read_csv(pitch_file, sep='\t', header=None, na_values=["--undefined--"]) \
            .iloc[1:-1][3].fillna(0.)
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


def prepare_strf_xy(out, feature_sets: list[str], subject_id: str,
                    fit_response_dimensions: Optional[list[int]] = None,
                    scaleflag=1, response_field="resp") -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, ...]]:
    response_field = response_field or 'resp'

    if not isinstance(scaleflag, np.ndarray) or scaleflag.ndim == 0:
        scaleflag = scaleflag * np.ones(len(feature_sets))

    if fit_response_dimensions is None:
        # TODO this should go in a config file somewhere
        if subject_id in ['EC53', 'EC85', 'EC89', 'EC143']:
            fit_response_dimensions = list(range(288))
        elif subject_id in ['EC212']:
            fit_response_dimensions = list(range(384))
        elif subject_id in ['EC75', 'EC124', 'EC200']:
            fit_response_dimensions = list(range(64))
        elif subject_id in ['EC177', 'EC161']:
            fit_response_dimensions = list(range(128))
        elif subject_id in ['EC193']:
            fit_response_dimensions = list(range(274))
        else:
            fit_response_dimensions = list(range(256))
    
    # estimation parameters
    inclSentOns = np.ones(len(feature_sets))
    zscoreXflag = np.ones(len(feature_sets))
    zscoreYflag = np.zeros(len(feature_sets))
    scaleXflag = np.ones(len(feature_sets))
    scaleYflag = np.zeros(len(feature_sets))
    highpassflag = np.zeros(len(feature_sets))
    modelnames = [f"{feature_sets[i]}_zX{zscoreXflag[i]}_zY{zscoreYflag[i]}_scX{scaleXflag[i]}_scY{scaleYflag[i]}_hp{highpassflag[i]}_SentOns{inclSentOns[i]}_sentScale{scaleflag[i]}"
                  for i in range(len(feature_sets))]
    
    # STRF setup - do not change these parameters
    time_lag = 0.6
    regalphas = np.logspace(0, 7, 20)

    # paper strfs
    nfold = 5;
    fullmodel_flag = 0;
    bootflag = 1;
    nboots = 10;
    edgeflag = 1;
    dataf = out[0]["dataf"];
    fs = out[0]["soundf"]
    # assert dataf == fs, f"Neural and sound sampling rate must match ({dataf} Hz != {fs} Hz)"

    X, Y, feature_names, feature_shapes = [], [], None, None
    for i, trial in enumerate(out):
        # TODO what to do about repeats?
        Yi = trial[response_field]
        if Yi.ndim == 3:
            # TODO ?
            continue
    
        Xi = []
        feature_names_i, feature_shape_i = [], []
        for feature_lookup in feature_sets:
            feature = trial[feature_lookup]
            if feature.ndim == 2:
                for j, feature_row in enumerate(feature):
                    Xi.append(feature_row)

                    feature_names_i.append(f"{feature_lookup}_{j}")
                feature_shape_i.append(feature.shape[0])
            else:
                Xi.append(feature)
                feature_names_i.append(feature_lookup)
                feature_shape_i.append(1)

        # print(list(zip([xi.shape for xi in Xi], feature_names_i)))
        Xi = np.stack(Xi).astype(float).T

        # Remove padding
        padding = tuple(trial.get("befaft", (0.5, 0.5)))
        start_sample = int(padding[0] * dataf)
        end_sample = Xi.shape[0] - int(padding[1] * dataf)

        Xi = Xi[start_sample:end_sample]
        Yi = Yi[start_sample:end_sample]

        X.append(Xi)
        Y.append(Yi)

        if feature_names is None:
            feature_names = feature_names_i
            feature_shapes = feature_shape_i
        else:
            assert feature_names == feature_names_i
            assert feature_shapes == feature_shape_i

    # TODO sound onset -- strf_makeXtimeLag:79
    # TODO boundaries
    X = np.concatenate(X)
    Y = np.concatenate(Y, axis=1).T
    
    X -= X.mean(axis=0)
    X /= (X.max(axis=0) - X.min(axis=0))

    # Subset instructed electrodes
    if fit_response_dimensions is not None:
        Y = Y[:, fit_response_dimensions]
    # Normalize response
    Y -= Y.mean(axis=0)
    Y /= (Y.max(axis=0) - Y.min(axis=0))

    return X, Y, feature_names, feature_shapes


def concat_xy(all_xy: list[tuple[np.ndarray, np.ndarray, list[str], tuple[int, ...]]]):
    X, Y, feature_names, feature_shapes = [], [], None, None
    for Xi, Yi, feature_names_i, feature_shapes_i in all_xy:
        X.append(Xi)
        Y.append(Yi)

        if feature_names is None:
            feature_names = feature_names_i
            feature_shapes = feature_shapes_i
        else:
            assert feature_names == feature_names_i
            assert feature_shapes == feature_shapes_i

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    # TODO handle boundaries

    return X, Y, feature_names, feature_shapes


def strf_nested_cv(X, Y, feature_names, feature_shapes, sfreq,
                   trf_kwargs=None,
                   hparams=None,
                   cv_outer=None, cv_inner=None,
                   ) -> tuple[TemporalReceptiveField,
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
        "fit_intercept": False,
        "verbose": False,
    }
    trf_kwargs = {**default_trf_kwargs, **(trf_kwargs or {})}
    regression = TemporalReceptiveField(
        sfreq=sfreq,
        feature_names=feature_names,
        **trf_kwargs)

    if cv_outer is None or isinstance(cv_outer, int):
        cv_outer = KFold(n_splits=cv_outer or 5, shuffle=False)
    if cv_inner is None or isinstance(cv_inner, int):
        cv_inner = KFold(n_splits=cv_inner or 5, shuffle=False)

    if hparams is None:
        hparams = {'estimator': np.logspace(0, 7, 5)}

    preds, scores, coefs, best_hparams = [], [], [], []
    for train, test in tqdm(cv_outer.split(X, Y), total=cv_outer.get_n_splits()):    
        model = GridSearchCV(regression, hparams,
                             cv=cv_inner,
                             verbose=1,
                             n_jobs=4)
        model.fit(X[train], Y[train])

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

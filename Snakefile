import itertools
import sys
from typing import Any
import yaml

import papermill

configfile: "config.yaml"

include: "workflows/librispeech/Snakefile"

ruleorder:
    run_no_train > run

wildcard_constraints:
    dataset = r"[a-z0-9_-]+",
#     feature_sets = r"[a-z0-9_]+",
#     subject = r"[a-z0-9_]+",


# Notebooks to run for intrinsic analysis on models
ALL_MODEL_NOTEBOOKS = [
    "lexical_coherence",
    "lexical_coherence_for_aggregation",
    "line_search",
    "syllable_coherence",
    "syllable_coherence_by_position",
    "phoneme_coherence",
    "phoneme_coherence_by_position",
    "temporal_generalization_word",
    "temporal_generalization_phoneme",
    "predictions",
    "predictions_word",
    "rsa_phoneme",
    "state_space",
    "trf",
    "within_word_gradience",
    "word_discrimination",

    "word_boundary",
    "syllable_boundary",

    "geometry/analogy",
    "geometry/analogy_dynamic",
]


ALL_ENCODING_SUBJECTS = [data_spec['subject'] for data_spec in config["encoding"]["data"]]
ENCODING_DATASET = "timit-no_repeats"

DEFAULT_PHONEME_EQUIVALENCE = "phoneme_10frames"

FOMO_LIBRARY_PATH = "/userdata/jgauthier/projects/neural-foundation-models:/userdata/jgauthier/projects/neural-foundation-models/data_utils"
sys.path.append(FOMO_LIBRARY_PATH)


def select_gpu_device(wildcards, resources):
    if resources.gpu == 0:
        return None
    import GPUtil
    available_l = GPUtil.getAvailable(
        order = 'random', limit = resources.gpu,
        maxLoad = 0.01, maxMemory = 0.4, includeNan=False,
        excludeID=[], excludeUUID=[])
    available_str = ",".join([str(x) for x in available_l])

    if len(available_l) == 0 and resources.gpu > 0:
        raise Exception("select_gpu_device did not select any GPUs")
    elif len(available_l) < resources.gpu:
        sys.stderr.write("[WARN] select_gpu_device selected fewer GPU devices than requested")
    print("Assigning %d available GPU devices: %s" % (resources.gpu, available_str))
    return available_str


def hydra_param(obj):
    """
    Prepare the given object for use as a Hydra CLI / YAML override.
    """
    if isinstance(obj, snakemake.io.Namedlist):
        obj = list(obj)
    return yaml.safe_dump(obj, default_flow_style=True, width=float("inf")).strip()

def join_hydra_overrides(overrides: dict[str, Any]):
    return " ".join([f"+{k}={v}" for k, v in overrides.items()])


rule preprocess:
    input:
        timit_raw = lambda wildcards: config["datasets"][wildcards.dataset]["raw_path"],
        script = "notebooks/preprocessing/timit.ipynb"

    output:
        data_path = directory("outputs/preprocessed_data/{dataset}"),
        notebook_path = "outputs/preprocessing/{dataset}/timit.ipynb"

    shell:
        """
        papermill --log-output {input.script} \
            {output.notebook_path} \
            -p base_dir {workflow.basedir} \
            -p dataset_path {input.timit_raw} \
            -p dataset_name {wildcards.dataset} \
            -p out_path {output.data_path}
        """


rule extract_hidden_states:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        base_model_config = "conf/base_model/{base_model_name}.yaml"

    resources:
        gpu = 1

    output:
        "outputs/hidden_states/{base_model_name}/{dataset}.h5"

    run:
        outdir = Path(output[0]).parent
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python scripts/extract_hidden_states.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={output} \
            dataset.processed_data_dir={input.dataset}
        """)


rule prepare_equivalence_dataset:
    input:
        timit_data = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml"

    output:
        "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    run:
        outdir = Path(output[0]).parent

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        python scripts/make_equivalence_dataset.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={wildcards.equivalence_classer} \
            dataset.processed_data_dir={input.timit_data}
        """)


def _get_equivalence_dataset(dataset: str, base_model_name: str, equivalence_classer: str) -> str:
    if equivalence_classer == "random":
        # default to phoneme-level
        return f"outputs/equivalence_datasets/{dataset}/{base_model_name}/{DEFAULT_PHONEME_EQUIVALENCE}/equivalence.pkl"
    else:
        return f"outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"


def get_equivalence_dataset(wildcards):
    # if we have a target_dataset, we should be retrieving equivalences for there
    return _get_equivalence_dataset(getattr(wildcards, "target_dataset", wildcards.dataset),
                                    wildcards.base_model_name, wildcards.equivalence_classer)


rule run:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml",
        model_config = "conf/model/{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    resources:
        gpu = 1

    output:
        full_trace = directory("outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}")

    run:
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export CUDA_VISIBLE_DEVICES={gpu_device}
        export HDF5_USE_FILE_LOCKING=FALSE
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            dataset.processed_data_dir={input.dataset} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            model={wildcards.model_name} \
            equivalence={wildcards.equivalence_classer} \
            +equivalence.path={input.equivalence_dataset}
        """)


rule tune_hparam:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml",
        model_config = "conf/model/{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    resources:
        gpu = 1

    output:
        full_trace = directory("outputs/hparam_search/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}")

    run:
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export CUDA_VISIBLE_DEVICES={gpu_device}
        export HDF5_USE_FILE_LOCKING=FALSE
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            trainer.mode=hyperparameter_search \
            dataset.processed_data_dir={input.dataset} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            model={wildcards.model_name} \
            equivalence={wildcards.equivalence_classer} \
            +equivalence.path={input.equivalence_dataset}
        """)


# Run train without actually training -- used to generate random model weights
NO_TRAIN_DEFAULT_EQUIVALENCE = DEFAULT_PHONEME_EQUIVALENCE
rule run_no_train:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = f"conf/equivalence/{NO_TRAIN_DEFAULT_EQUIVALENCE}.yaml",
        model_config = "conf/model/random{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_dataset = f"outputs/equivalence_datasets/{{dataset}}/{{base_model_name}}/{NO_TRAIN_DEFAULT_EQUIVALENCE}/equivalence.pkl"

    output:
        full_trace = directory("outputs/models/{dataset}/{base_model_name}/random{model_name}/random")

    shell:
        """
        export HDF5_USE_FILE_LOCKING=FALSE
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            dataset.processed_data_dir={input.dataset} \
            model=random{wildcards.model_name} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={NO_TRAIN_DEFAULT_EQUIVALENCE} \
            +equivalence.path={input.equivalence_dataset} \
            trainer.mode=no_train \
            device=cpu
        """


MODEL_SPEC_LIST = [f"{m['dataset']}/{m['base_model']}/{m['model']}/{m['equivalence']}" for m in config["models"]]
rule run_all:
    input:
        expand("outputs/models/{model_spec}", model_spec=MODEL_SPEC_LIST)


rule extract_embeddings:
    input:
        model_dir = "outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{target_dataset}.h5",
        equivalence_dataset = get_equivalence_dataset

    resources:
        gpu = 1

    output:
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{target_dataset}.npy"

    run:
        outdir = Path(output.embeddings).parent
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python scripts/extract_model_embeddings.py \
            hydra.run.dir={outdir} \
            model={wildcards.model_name} \
            +model.output_dir={input.model_dir} \
            +model.embeddings_path={output.embeddings} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={wildcards.equivalence_classer} \
            +equivalence.path={input.equivalence_dataset}
        """)



# rule extract_all_embeddings:
#     input:
#         expand("outputs/model_embeddings/{model_spec}/embeddings.npy",
#                 model_spec=MODEL_SPEC_LIST)

rule compute_state_spaces:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5"

    output:
        "outputs/state_space_specs/{dataset}/{base_model_name}/state_space_specs.h5"

    run:
        outdir = Path(output[0]).parent

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        python scripts/generate_state_space_specs.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            dataset.processed_data_dir={input.dataset} \
            +base_model.hidden_state_path={input.hidden_states} \
            +analysis.state_space_specs_path={output[0]}
        """)


NOTEBOOK_PHONEME_EQUIVALENCE = "phoneme_10frames"
NOTEBOOK_WORD_EQUIVALENCE = "word_broad_10frames"
rule run_notebook:
    input:
        notebook = "notebooks/{notebook}.ipynb",
        model_dir = "outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}",

        dataset = "outputs/preprocessed_data/{dataset}",

        # reference particular equivalence dataset of this model, as well as standard
        # equivalence representations at different unit levels
        equivalence_dataset = get_equivalence_dataset,
        phoneme_equivalence_dataset = f"outputs/equivalence_datasets/{{dataset}}/{{base_model_name}}/{NOTEBOOK_PHONEME_EQUIVALENCE}/equivalence.pkl",
        word_equivalence_dataset = f"outputs/equivalence_datasets/{{dataset}}/{{base_model_name}}/{NOTEBOOK_WORD_EQUIVALENCE}/equivalence.pkl",

        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        state_space_specs = "outputs/state_space_specs/{dataset}/{base_model_name}/state_space_specs.h5",
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{dataset}.npy"

    output:
        outdir = directory("outputs/notebooks/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{notebook}"),
        notebook = "outputs/notebooks/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{notebook}/{notebook}.ipynb"

    shell:
        """
        export HDF5_USE_FILE_LOCKING=FALSE
        papermill --autosave-cell-every 30 --log-output \
            {input.notebook} {output.notebook} \
            -p model_dir {input.model_dir} \
            -p output_dir {output.outdir} \
            -p dataset_path {input.dataset} \
            -p equivalence_path {input.equivalence_dataset} \
            -p phoneme_equivalence_path {input.phoneme_equivalence_dataset} \
            -p word_equivalence_path {input.word_equivalence_dataset} \
            -p hidden_states_path {input.hidden_states} \
            -p state_space_specs_path {input.state_space_specs} \
            -p embeddings_path {input.embeddings}
        """


rule run_all_notebooks:
    input:
        expand("outputs/notebooks/{model_spec}/{notebook}/{notebook}.ipynb",
                model_spec=MODEL_SPEC_LIST, notebook=ALL_MODEL_NOTEBOOKS)


rule evaluate_word_recognition:
    input:
        model_dir = "outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{target_dataset}.h5",
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{target_dataset}.npy",
        state_space_specs = "outputs/state_space_specs/{dataset}/{base_model_name}/state_space_specs.h5",

        model_config = "conf/recognition_model/{recognition_model}.yaml"

    resources:
        gpu = 1

    output:
        trace = directory("outputs/word_recognition/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{target_dataset}/{recognition_model}"),

    run:
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python word_recognition.py \
            hydra.run.dir={output.trace} \
            recognition_model={wildcards.recognition_model} \
            model={wildcards.model_name} \
            +model.output_dir={input.model_dir} \
            +model.embeddings_path={input.embeddings} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            +analysis.state_space_specs_path={input.state_space_specs} \
        """)


rule run_all_word_recognition:
    input:
        expand("outputs/word_recognition/{model_spec}/librispeech-train-clean-100/linear",
                model_spec=MODEL_SPEC_LIST)


def _get_inputs_for_encoding(feature_set: str, encoding_dataset: str, return_list=False) -> list[str]:
    """
    Args:
        feature_set: Reference to a feature set in the config (`conf_encoder/feature_sets`)
        encoding_dataset: Target dataset for encoding
        return_list:

    Returns:
        A dict (or flattened list if `return_list`) of paths to the inputs required for encoding with the given feature set.
    """
    with open(f"conf_encoder/feature_sets/{feature_set}.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    # The dataset on which the model was trained may be different than the target dataset
    input_specs = {}
    for key, feature_set in model_config.get("model_features", {}).items():
        train_dataset = feature_set.get("train_dataset", encoding_dataset)
        input_specs[key] = {
            "hidden_states": f"outputs/hidden_states/{feature_set['base_model']}/{encoding_dataset}.h5",
            "equivalence": _get_equivalence_dataset(encoding_dataset, feature_set['base_model'], feature_set['equivalence']),
            "embeddings": f"outputs/model_embeddings/{train_dataset}/{feature_set['base_model']}/{feature_set['model']}/{feature_set['equivalence']}/{encoding_dataset}.npy",
            "state_space": f"outputs/state_space_specs/{encoding_dataset}/{feature_set['base_model']}/state_space_specs.h5"
        }

    # restructure to return a list of paths for each type of input
    ret = {
        "hidden_states": [v["hidden_states"] for v in input_specs.values()],
        "equivalences": [v["equivalence"] for v in input_specs.values()],
        "state_spaces": [v["state_space"] for v in input_specs.values()],
        "embeddings": [v["embeddings"] for v in input_specs.values()],
    }

    if return_list:
        return list(itertools.chain.from_iterable(ret.values()))
    else:
        return ret


rule estimate_synthetic_encoder:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{target_model_name}/{dataset}.h5",
        computed_inputs = lambda wildcards: itertools.chain.from_iterable(
            _get_inputs_for_encoding(feature_set, wildcards.dataset, return_list=True)
            for feature_set in config["synthetic_encoding"]["evaluations"][wildcards.evaluation_name]["models"]
        ),
        notebook = "notebooks/synthetic_encoding/feature_selection.ipynb"

    output:
        model_dir = directory("outputs/synthetic_encoders/{dataset}/{target_model_name}/{evaluation_name}/{subsample_strategy}"),
        notebook = "outputs/synthetic_encoders/{dataset}/{target_model_name}/{evaluation_name}/{subsample_strategy}/feature_selection.ipynb",

    run:
        evaluation = config["synthetic_encoding"]["evaluations"][wildcards.evaluation_name]

        # Recompute embedding paths with keys here
        embedding_paths = {}
        for model_name in evaluation["models"]:
            encoding_inputs = _get_inputs_for_encoding(model_name, wildcards.dataset)
            assert len(encoding_inputs["embeddings"]) == 1
            embedding_paths[model_name] = encoding_inputs["embeddings"][0]

        # They all share the same state space
        state_space_path = encoding_inputs["state_spaces"][0]

        params = {
            "dataset_path": input.dataset,
            "hidden_states_path": input.hidden_states,
            "state_space_specs_path": state_space_path,
            "output_dir": output.model_dir,
            "model_embedding_paths": embedding_paths,

            "target_smoosh": evaluation.get("target_smoosh", None),
            "num_components": evaluation["num_components"],
            "num_embeddings_to_select": evaluation["num_embeddings_to_select"],
        }

        shell(f"""
        export HDF5_USE_FILE_LOCKING=FALSE
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -y "{yaml.safe_dump(params)}"
        """)

rule estimate_all_synthetic_encoders:
    input:
        lambda wildcards: [f"outputs/synthetic_encoders/{dataset}/{target_model}/{evaluation_name}/{subsample_strategy}"
                           for evaluation_name, evaluation in config["synthetic_encoding"]["evaluations"].items()
                           for dataset in evaluation["datasets"]
                           for target_model in evaluation["target_models"]
                           for subsample_strategy in evaluation["subsample_strategies"]]


def make_encoder_data_spec(include_subjects=None):
    data_spec = config["encoding"]["data"]
    if include_subjects is not None:
        data_spec = [d for d in data_spec if d["subject"] in include_subjects]
        if len(data_spec) == 0:
            raise ValueError(f"No data for subjects {include_subjects}")

    # Now expand by block
    data_spec = [{"block": block,
                  **{k: v for k, v in d.items() if k != "blocks"}}
                 for d in data_spec
                 for block in (d.get("blocks") if d.get("blocks") is not None else [None])]

    return hydra_param(data_spec)


def run_encoder(input, output, wildcards, overrides=None):
    encoder_inputs = _get_inputs_for_encoding(wildcards.feature_sets, wildcards.dataset)
    assert len(encoder_inputs["embeddings"]) == len(encoder_inputs["equivalences"]) \
        == len(encoder_inputs["hidden_states"])

    data_spec = make_encoder_data_spec(include_subjects=[wildcards.subject])
    if len(data_spec) == 0:
        raise ValueError(f"No data for subject {wildcards.subject}")

    # Prepare overrides for each feature set's inputs (equivalence dataset and embedding)
    local_overrides = {}
    if len(encoder_inputs["embeddings"]) > 0:
        for feature_set, embedding_path, equivalence_path, hidden_state_path, state_space_path in zip(
                [wildcards.feature_sets], encoder_inputs["embeddings"],
                encoder_inputs["equivalences"], encoder_inputs["hidden_states"],
                encoder_inputs["state_spaces"]):
            local_overrides[f"feature_sets.model_features.{feature_set}.embeddings_path"] = embedding_path
            local_overrides[f"feature_sets.model_features.{feature_set}.equivalence_path"] = equivalence_path
            local_overrides[f"feature_sets.model_features.{feature_set}.hidden_state_path"] = hidden_state_path
            local_overrides[f"feature_sets.model_features.{feature_set}.state_space_path"] = state_space_path

    overrides = overrides if overrides is not None else {}
    assert not (set(local_overrides.keys()) & overrides.keys()), "Local overrides conflict with global overrides"
    overrides_str = join_hydra_overrides({**local_overrides, **overrides})

    shell("""
    # add fomo to path
    export PYTHONPATH={FOMO_LIBRARY_PATH}:${{PYTHONPATH:-}}
    python estimate_encoder.py \
        hydra.run.dir={output.model_dir} \
        feature_sets={wildcards.feature_sets} \
        +dataset_path={input.dataset} \
        {overrides_str} \
        +data='{data_spec}'
    """)


rule estimate_noise_ceiling:
    output:
        "outputs/encoder_noise_ceiling/splithalf_corrs.csv"

    run:
        outdir = Path(output[0]).parent
        data_spec = make_encoder_data_spec()

        shell("""
        export PYTHONPATH=.
        python scripts/estimate_noise_ceiling_splithalf.py \
            hydra.run.dir={outdir} \
            +data='{data_spec}'
        """)


rule estimate_encoder:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        encoder_config = "conf_encoder/feature_sets/{feature_sets}.yaml",
        computed_inputs = lambda wildcards: _get_inputs_for_encoding(
            wildcards.feature_sets, wildcards.dataset, return_list=True)
    output:
        model_dir = directory("outputs/encoders/{dataset}/{feature_sets}/{subject}"),
        electrodes = "outputs/encoders/{dataset}/{feature_sets}/{subject}/electrodes.csv",
        scores = "outputs/encoders/{dataset}/{feature_sets}/{subject}/scores.csv",
        predictions = "outputs/encoders/{dataset}/{feature_sets}/{subject}/predictions.npy",
        model = "outputs/encoders/{dataset}/{feature_sets}/{subject}/model.pkl",
        coefs = "outputs/encoders/{dataset}/{feature_sets}/{subject}/coefs.pkl",
        hparams = "outputs/encoders/{dataset}/{feature_sets}/{subject}/hparams.pkl"

    run:
        run_encoder(input, output, wildcards)


all_encoding_outputs = lambda wildcards: \
    set(f"outputs/encoders/{ENCODING_DATASET}/{comp['model2']}/{subject}"
        for comp in config["encoding"]["model_comparisons"]
        for subject in ALL_ENCODING_SUBJECTS) | \
    set(f"outputs/encoders/{ENCODING_DATASET}/{comp['model1']}/{subject}"
        for comp in config["encoding"]["model_comparisons"]
        for subject in ALL_ENCODING_SUBJECTS)


"""
Estimate a single permutation baseline for a given model embedding.
"""
rule estimate_encoder_unit_permutation:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        encoder_config = "conf_encoder/feature_sets/{feature_sets}.yaml",
        computed_inputs = lambda wildcards: _get_inputs_for_encoding(
            wildcards.feature_sets, wildcards.dataset, return_list=True)
    output:
        model_dir = directory("outputs/encoders-permute_{permutation_name}/{permutation_idx}/{dataset}/{feature_sets}/{subject}"),
        electrodes = "outputs/encoders-permute_{permutation_name}/{permutation_idx}/{dataset}/{feature_sets}/{subject}/electrodes.csv",
        scores = "outputs/encoders-permute_{permutation_name}/{permutation_idx}/{dataset}/{feature_sets}/{subject}/scores.csv",
        predictions = "outputs/encoders-permute_{permutation_name}/{permutation_idx}/{dataset}/{feature_sets}/{subject}/predictions.npy",
        model = "outputs/encoders-permute_{permutation_name}/{permutation_idx}/{dataset}/{feature_sets}/{subject}/model.pkl",
        coefs = "outputs/encoders-permute_{permutation_name}/{permutation_idx}/{dataset}/{feature_sets}/{subject}/coefs.pkl"

    run:
        permutation_type = config["encoding"]["permutation_tests"][wildcards.permutation_name]["permutation"]
        overrides = {
            f"feature_sets.model_features.{wildcards.feature_sets}.permute": permutation_type,
        }
        run_encoder(input, output, wildcards, overrides=overrides)


all_permutation_outputs = lambda wildcards: [f"outputs/encoders-permute_{perm_name}/{perm_idx}/{ENCODING_DATASET}/{comp['model2']}/{subject}"
                           for comp in config["encoding"]["model_comparisons"]
                           for subject in ALL_ENCODING_SUBJECTS
                           for perm_name, perm in config["encoding"]["permutation_tests"].items()
                           for perm_idx in range(perm["num_permutations"])]
rule estimate_all_permutations:
    input:
        all_permutation_outputs


rule estimate_encoder_unique_variance:
    input:
        encoder = "outputs/encoders/{dataset}/{feature_sets}/{subject}",
        notebook = "notebooks/encoding/unique_variance.ipynb",

    output:
        dir = directory("outputs/encoder_unique_variance/{dataset}/{feature_sets}/{subject}"),
        notebook = "outputs/encoder_unique_variance/{dataset}/{feature_sets}/{subject}/unique_variance.ipynb",
        unique_variance = "outputs/encoder_unique_variance/{dataset}/{feature_sets}/{subject}/unique_variance.csv"

    run:
        output_dir = Path(output.unique_variance).parent

        shell("""
        export PYTHONPATH="`pwd`:{FOMO_LIBRARY_PATH}"
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -p encoder_path {input.encoder} \
            -p output_dir {output_dir}
        """)


all_unique_variance_outputs = expand(f"outputs/encoder_unique_variance/{ENCODING_DATASET}/baseline/{{subject}}/unique_variance.csv",
               subject=ALL_ENCODING_SUBJECTS)
rule estimate_all_baseline_unique_variance:
    input:
        all_unique_variance_outputs


all_encoder_within_subject_comparisons = lambda _: [
    f"outputs/encoder_comparison/{ENCODING_DATASET}/{subject}/{comp['model2']}/{comp['model1']}/"
    for comp in config["encoding"]["model_comparisons"] for subject in ALL_ENCODING_SUBJECTS]

rule compare_encoder_within_subject:
    input:
        model1_scores = "outputs/encoders/{dataset}/{comparison_model1}/{subject}/scores.csv",
        model2_scores = "outputs/encoders/{dataset}/{comparison_model2}/{subject}/scores.csv",

        model1_model = "outputs/encoders/{dataset}/{comparison_model1}/{subject}/model.pkl",
        model2_model = "outputs/encoders/{dataset}/{comparison_model2}/{subject}/model.pkl",

        model1_coefs = "outputs/encoders/{dataset}/{comparison_model1}/{subject}/coefs.pkl",
        model2_coefs = "outputs/encoders/{dataset}/{comparison_model2}/{subject}/coefs.pkl",

        model2_permutation_scores = lambda wildcards: [
            f"outputs/encoders-permute_{perm_name}/{perm_idx}/{wildcards.dataset}/{wildcards.comparison_model2}/{wildcards.subject}/scores.csv"
            for perm_name, perm in config["encoding"]["permutation_tests"].items()
            for perm_idx in range(perm["num_permutations"])
        ],

        notebook = "notebooks/encoding/compare_within_subject.ipynb",

    output:
        comparison_dir = directory("outputs/encoder_comparison/{dataset}/{subject}/{comparison_model2}/{comparison_model1}"),
        notebook = "outputs/encoder_comparison/{dataset}/{subject}/{comparison_model2}/{comparison_model1}/comparison.ipynb",
        scores_csv = "outputs/encoder_comparison/{dataset}/{subject}/{comparison_model2}/{comparison_model1}/scores.csv",
        improvements_csv = "outputs/encoder_comparison/{dataset}/{subject}/{comparison_model2}/{comparison_model1}/improvements.csv",
        permutation_improvements_csv = "outputs/encoder_comparison/{dataset}/{subject}/{comparison_model2}/{comparison_model1}/permutation_improvements.csv",
        ttest_results = "outputs/encoder_comparison/{dataset}/{subject}/{comparison_model2}/{comparison_model1}/ttest_results.csv",

    run:
        # group permutation scores by permutation test
        permutation_scores = {
            perm_name: [
                f"outputs/encoders-permute_{perm_name}/{perm_idx}/{wildcards.dataset}/{wildcards.comparison_model2}/{wildcards.subject}/scores.csv"
                for perm_idx in range(perm["num_permutations"])
            ]
            for perm_name, perm in config["encoding"]["permutation_tests"].items()
        }

        params = {
            "subject": wildcards.subject,
            "model1": wildcards.comparison_model1,
            "model2": wildcards.comparison_model2,
            "model1_scores_path": input.model1_scores,
            "model2_scores_path": input.model2_scores,
            "model1_coefs_path": input.model1_coefs,
            "model2_coefs_path": input.model2_coefs,
            "model1_model_path": input.model1_model,
            "model2_model_path": input.model2_model,
            "model2_permutation_score_paths": permutation_scores,
            "output_dir": output.comparison_dir,
        }
        shell(f"""
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -y "{yaml.safe_dump(params)}"
        """)


rule compare_all_encoders_within_subject:
    input:
        all_encoder_within_subject_comparisons,


rule compare_all_encoders_across_subject:
    input:
        all_comparisons = all_encoder_within_subject_comparisons,
        notebook = "notebooks/encoding/compare_across_subject.ipynb",

    output:
        dir = directory("outputs/encoder_comparison_across_subjects/{dataset}"),
        notebook = "outputs/encoder_comparison_across_subjects/{dataset}/notebook.ipynb",

        electrodes = "outputs/encoder_comparison_across_subjects/{dataset}/electrodes.csv",
        scores = "outputs/encoder_comparison_across_subjects/{dataset}/scores.csv",

        ttest = "outputs/encoder_comparison_across_subjects/{dataset}/ttest.csv",
        ttest_filtered = "outputs/encoder_comparison_across_subjects/{dataset}/ttest_filtered.csv",
    
    run:
        params = {
            "output_dir": output.dir,
            "encoder_comparisons": list(map(str, input.all_comparisons)),
        }
        
        shell(f"""
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -y "{yaml.safe_dump(params)}"
        """)


rule electrode_contrast:
    input:
        notebook = "notebooks/encoding/electrode_contrast.ipynb",
        ttest_results = "outputs/encoder_comparison_across_subjects/{dataset}/ttest.csv",
        scores = "outputs/encoder_comparison_across_subjects/{dataset}/scores.csv",
        encoder_dirs = all_encoding_outputs,
    
    output:
        dir = directory("outputs/electrode_contrast/{dataset}"),
        notebook = "outputs/electrode_contrast/{dataset}/electrode_contrast.ipynb",
        contrasts = "outputs/electrode_contrast/{dataset}/contrasts.csv",

    run:
        params = {
            "dataset": wildcards.dataset,
            "ttest_results_path": input.ttest_results,
            "scores_path": input.scores,
            "encoder_dirs": list(map(str, input.encoder_dirs)),
            "output_dir": output.dir,
        }
        shell(f"""
        export PYTHONPATH=`pwd`
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -y "{yaml.safe_dump(params)}"
        """)


rule electrode_study_within_subject:
    input:
        ttest_results = "outputs/encoder_comparison_across_subjects/{dataset}/ttest.csv",
        scores = "outputs/encoder_comparison_across_subjects/{dataset}/scores.csv",
        unique_variance = "outputs/encoder_unique_variance/{dataset}/baseline/{subject}/unique_variance.csv",
        notebook = "notebooks/encoding/electrode_study_within_subject.ipynb",
        contrasts = "outputs/electrode_contrast/{dataset}/contrasts.csv",
        encoder_dirs = lambda wildcards: [output for output in all_encoding_outputs(wildcards)
                                          if wildcards.subject in output],

    output:
        dir = directory("outputs/electrode_study/{dataset}/{subject}"),
        notebook = "outputs/electrode_study/{dataset}/{subject}/notebook.ipynb"

    run:
        params = {
            "dataset": wildcards.dataset,
            "subject": wildcards.subject,
            "ttest_results_path": input.ttest_results,
            "scores_path": input.scores,
            "contrasts_path": input.contrasts,
            "unique_variance_path": input.unique_variance,
            "encoder_dirs": list(map(str, input.encoder_dirs)),
            "output_dir": output.dir,
        }
        papermill.execute_notebook(
            input.notebook, output.notebook,
            parameters=params,
            log_output=True)


all_electrode_study_outputs = lambda _: [f"outputs/electrode_study/{ENCODING_DATASET}/{subject}/"
                   for subject in ALL_ENCODING_SUBJECTS],
rule electrode_study_within_subject_all:
    input:
        all_electrode_study_outputs,


rule encoding_sanity_checks:
    input:
        notebook = "notebooks/encoding/sanity_checks.ipynb",
        ttest_results = f"outputs/encoder_comparison_across_subjects/{ENCODING_DATASET}/ttest.csv",
        score_results = f"outputs/encoder_comparison_across_subjects/{ENCODING_DATASET}/scores.csv",
        all_encoding_outputs = all_encoding_outputs,
        all_permutation_outputs = all_permutation_outputs,

    output:
        outdir = directory(f"outputs/encoding_sanity_checks/{ENCODING_DATASET}"),
        notebook = f"outputs/encoding_sanity_checks/{ENCODING_DATASET}/sanity_checks.ipynb"

    run:
        params = {
            "dataset": ENCODING_DATASET,
            "encoder_dirs": list(map(str, input.all_encoding_outputs)),
            "permuted_encoder_dirs": list(map(str, input.all_permutation_outputs)),
            "ttest_results_path": input.ttest_results,
            "scores_path": input.score_results,
            "output_dir": output.outdir,
        }

        shell(f"""
        export PYTHONPATH=`pwd`
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -y "{yaml.safe_dump(params)}"
        """)


# get all the encoding results
rule encoding:
    input:
        all_electrode_study_outputs,
        f"outputs/encoder_comparison_across_subjects/{ENCODING_DATASET}/ttest_filtered.csv",
        f"outputs/electrode_contrast/{ENCODING_DATASET}/contrasts.csv",
        all_encoding_outputs,
        all_permutation_outputs,
        all_unique_variance_outputs,
        f"outputs/encoding_sanity_checks/{ENCODING_DATASET}/sanity_checks.ipynb",


rule estimate_rsa:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        computed_inputs = lambda wildcards: _get_inputs_for_encoding(
            wildcards.feature_sets, wildcards.dataset, return_list=True)
    output:
        model_dir = directory("outputs/rsa/{dataset}/{analysis}/{feature_sets}/{state_space}/{subject}"),
        electrodes = "outputs/rsa/{dataset}/{analysis}/{feature_sets}/{state_space}/{subject}/model_electrode_dists.csv"

    run:
        encoder_inputs = _get_inputs_for_encoding(wildcards.feature_sets, wildcards.dataset)
        assert len(encoder_inputs["embeddings"]) == len(encoder_inputs["equivalences"]) \
            == len(encoder_inputs["hidden_states"])

        data_spec = make_encoder_data_spec(include_subjects=[wildcards.subject])
        if len(data_spec) == 0:
            raise ValueError(f"No data for subject {wildcards.subject}")

        # Prepare overrides for each feature set's inputs (equivalence dataset and embedding)
        overrides = {}
        if len(encoder_inputs["embeddings"]) > 0:
            for feature_set, embedding_path, equivalence_path, hidden_state_path, state_space_path in zip(
                    [wildcards.feature_sets], encoder_inputs["embeddings"],
                    encoder_inputs["equivalences"], encoder_inputs["hidden_states"],
                    encoder_inputs["state_spaces"]):
                overrides[f"feature_sets.model_features.{feature_set}.embeddings_path"] = embedding_path
                overrides[f"feature_sets.model_features.{feature_set}.equivalence_path"] = equivalence_path
                overrides[f"feature_sets.model_features.{feature_set}.hidden_state_path"] = hidden_state_path
                overrides[f"feature_sets.model_features.{feature_set}.state_space_path"] = state_space_path
        overrides_str = join_hydra_overrides(overrides)

        shell("""
        python rsa.py \
            hydra.run.dir={output.model_dir} \
            feature_sets={wildcards.feature_sets} \
            +dataset_path={input.dataset} \
            {overrides_str} \
            +data='{data_spec}' \
            +analysis={wildcards.analysis} \
            analysis.state_space={wildcards.state_space}
        """)
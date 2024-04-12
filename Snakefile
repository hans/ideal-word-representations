import itertools
from typing import Any
import yaml

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

configfile: "config.yaml"

ruleorder:
    run_no_train > run

wildcard_constraints:
    dataset = r"[a-z0-9_]+",
#     feature_sets = r"[a-z0-9_]+",
#     subject = r"[a-z0-9_]+",


# Notebooks to run for intrinsic analysis on models
ALL_MODEL_NOTEBOOKS = [
    "lexical_coherence",
    "syllable_coherence",
    "syllable_coherence_by_position",
    "phoneme_coherence",
    "phoneme_coherence_by_position",
    "temporal_generalization_word",
    "temporal_generalization_phoneme",
    "predictions",
    "predictions_word",
    "trf",
]


ALL_ENCODING_SUBJECTS = [data_spec['subject'] for data_spec in config["encoding"]["data"]]
ENCODING_DATASET = "timit"

DEFAULT_PHONEME_EQUIVALENCE = "phoneme_10frames"


def select_gpu_device(wildcards, resources):
    if resources.gpu == 0:
        return None
    import GPUtil
    available_l = GPUtil.getAvailable(
        order = 'random', limit = resources.gpu,
        maxLoad = 0.01, maxMemory = 0.01, includeNan=False,
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


# Download and convert fairseq wav2vec2 model pretrained on AudioSet data
# to a Huggingface model
rule prepare_audioset:
    input:
        checkpoint = HTTP.remote("https://huggingface.co/ewandunbar/humanlike-speech-2022/resolve/main/wav2vec/checkpoint_unsup_ac_scenes.pt"),
        run_forward_script = HTTP.remote("https://huggingface.co/HfSpeechUtils/convert_wav2vec2_to_hf/raw/main/run_forward.py")

    output:
        model_dir = directory("outputs/pretrained_models/ewandunbar_wav2vec2_humanlike-speech-2022_audioset"),
        config = "outputs/pretrained_models/ewandunbar_wav2vec2_humanlike-speech-2022_audioset/config.json",
        feature_extractor_config = "outputs/pretrained_models/ewandunbar_wav2vec2_humanlike-speech-2022_audioset/preprocessor_config.json",
        model_checkpoint = "outputs/pretrained_models/ewandunbar_wav2vec2_humanlike-speech-2022_audioset/pytorch_model.bin"

    shell:
        """
        # Dump model config and feature extractor config for wav2vec2
        python -c "import transformers; config = transformers.Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base'); config.save_pretrained('{output.model_dir}')"
        python -c "import transformers; fe = transformers.Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base'); fe.save_pretrained('{output.model_dir}')"

        transformers_dir="$(python -c 'import transformers; from pathlib import Path; print(Path(transformers.__file__).parent)')"
        python "${{transformers_dir}}/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py" \
            --pytorch_dump_folder {output.model_dir} \
            --checkpoint_path {input.checkpoint} \
            --config_path {output.model_dir}/config.json \
            --not_finetuned

        # Validate with forward pass
        python {input.run_forward_script} {output.model_dir} {input.checkpoint} 0
        """


rule preprocess_timit:
    input:
        timit_raw = config["datasets"]["timit"]["raw_path"],
        script = "notebooks/preprocessing/timit.ipynb"

    output:
        data_path = directory("outputs/preprocessed_data/timit"),
        notebook_path = "outputs/preprocessing/timit.ipynb"

    shell:
        """
        papermill --log-output {input.script} \
            {output.notebook_path} \
            -p base_dir {workflow.basedir} \
            -p dataset_path {input.timit_raw} \
            -p out_path {output.data_path}
        """


rule extract_hidden_states:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        base_model_config = "conf/base_model/{base_model_name}.yaml"

    resources:
        gpu = 1

    output:
        "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl"

    run:
        outdir = Path(output[0]).parent
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python scripts/extract_hidden_states.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            dataset.processed_data_dir={input.dataset}
        """)


rule prepare_equivalence_dataset:
    input:
        timit_data = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml"

    output:
        "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    run:
        outdir = Path(output[0]).parent

        shell("""
        export PYTHONPATH=`pwd`
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
    return _get_equivalence_dataset(wildcards.dataset, wildcards.base_model_name, wildcards.equivalence_classer)


rule run:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml",
        model_config = "conf/model/{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl",
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    resources:
        gpu = 1

    output:
        full_trace = directory("outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}")

    run:
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
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
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl",
        equivalence_dataset = f"outputs/equivalence_datasets/{{dataset}}/{{base_model_name}}/{NO_TRAIN_DEFAULT_EQUIVALENCE}/equivalence.pkl"

    output:
        full_trace = directory("outputs/models/{dataset}/{base_model_name}/random{model_name}/random")

    shell:
        """
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            dataset.processed_data_dir={input.dataset} \
            model=random{wildcards.model_name} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={NO_TRAIN_DEFAULT_EQUIVALENCE} \
            +equivalence.path={input.equivalence_dataset} \
            trainer.do_train=false
        """


MODEL_SPEC_LIST = [f"{m['dataset']}/{m['base_model']}/{m['model']}/{m['equivalence']}" for m in config["models"]]
rule run_all:
    input:
        expand("outputs/models/{model_spec}", model_spec=MODEL_SPEC_LIST)


rule extract_embeddings:
    input:
        model_dir = "outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}",
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl",
        equivalence_dataset = get_equivalence_dataset

    resources:
        gpu = 1

    output:
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/embeddings.npy"

    run:
        outdir = Path(output.embeddings).parent
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
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


rule compute_state_spaces:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl"

    output:
        "outputs/state_space_specs/{dataset}/{base_model_name}/state_space_specs.pkl"

    run:
        outdir = Path(output[0]).parent

        shell("""
        export PYTHONPATH=`pwd`
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

        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl",
        state_space_specs = "outputs/state_space_specs/{dataset}/{base_model_name}/state_space_specs.pkl",
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/embeddings.npy"

    output:
        outdir = directory("outputs/notebooks/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{notebook}"),
        notebook = "outputs/notebooks/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{notebook}/{notebook}.ipynb"

    shell:
        """
        papermill --log-output \
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


def _get_inputs_for_encoding(feature_set: str, dataset: str, return_list=False) -> list[str]:
    with open(f"conf_encoder/feature_sets/{feature_set}.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    ret = {
        "hidden_states": [
            f"outputs/hidden_states/{dataset}/{feat['base_model']}/hidden_states.pkl"
            for _, feat in model_config.get("model_features", {}).items()
        ],

        "embeddings": [
            f"outputs/model_embeddings/{dataset}/{feat['base_model']}/{feat['model']}/{feat['equivalence']}/embeddings.npy"
            for _, feat in model_config.get("model_features", {}).items()
        ],

        "equivalences": [
            _get_equivalence_dataset(dataset, feat['base_model'], feat['equivalence'])
            for _, feat in model_config.get("model_features", {}).items()
        ],

        "state_spaces": [
            f"outputs/state_space_specs/{dataset}/{feat['base_model']}/state_space_specs.pkl"
            for _, feat in model_config.get("model_features", {}).items()
        ]
    }

    if return_list:
        return list(itertools.chain.from_iterable(ret.values()))
    else:
        return ret


rule estimate_synthetic_encoder:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{dataset}/{target_model_name}/hidden_states.pkl",
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

    # Now expand by block
    data_spec = [{"subject": d["subject"], "block": block}
                 for d in data_spec
                 for block in (d["blocks"] if d["blocks"] is not None else [None])]

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
        coefs = "outputs/encoders/{dataset}/{feature_sets}/{subject}/coefs.pkl"

    run:
        run_encoder(input, output, wildcards)


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


rule estimate_all_permutations:
    input:
        lambda wildcards: [f"outputs/encoders-permute_{perm_name}/{perm_idx}/{ENCODING_DATASET}/{comp['model2']}/{subject}"
                           for comp in config["encoding"]["model_comparisons"]
                           for subject in ALL_ENCODING_SUBJECTS
                           for perm_name, perm in config["encoding"]["permutation_tests"].items()
                           for perm_idx in range(perm["num_permutations"])]


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
        assert wildcards.dataset == ENCODING_DATASET
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


rule compare_all_encoders_across_subject:
    input:
        all_comparisons = lambda _: [
            f"outputs/encoder_comparison/{{dataset}}/{subject}/{comp['model2']}/{comp['model1']}/"
            for comp in config["encoding"]["model_comparisons"] for subject in ALL_ENCODING_SUBJECTS],
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


rule colocation_study_within_subject:
    input:
        encoder_ttest = "outputs/encoder_comparison_across_subjects/{dataset}/ttest_filtered.csv",
        notebook = "notebooks/encoding/colocation_within_subject.ipynb",
        encoders = lambda wildcards: [f"outputs/encoders/{{dataset}}/{comp['model2']}/{{subject}}/"
                                      for comp in config["encoding"]["model_comparisons"]]

    output:
        dir = directory("outputs/encoder_colocation_study/{dataset}/{subject}"),
        notebook = "outputs/encoder_colocation_study/{dataset}/{subject}/notebook.ipynb",

    run:
        params = {
            "dataset": wildcards.dataset,
            "subject": wildcards.subject,
            "ttest_results_path": input.encoder_ttest,
            "output_dir": output.dir,
            "encoder_dirs": list(map(str, input.encoders)),
        }

        shell(f"""
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -y "{yaml.safe_dump(params)}"
        """)


rule all_colocation_studies:
    input:
        lambda _: [f"outputs/encoder_colocation_study/{ENCODING_DATASET}/{subject}/"
                   for subject in ALL_ENCODING_SUBJECTS],


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
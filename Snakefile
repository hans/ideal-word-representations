import yaml

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

configfile: "config.yaml"

ruleorder:
    run_no_train > run


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


def hydra_param(obj):
    """
    Prepare the given object for use as a Hydra CLI / YAML override.
    """
    if isinstance(obj, snakemake.io.Namedlist):
        obj = list(obj)
    return yaml.safe_dump(obj, default_flow_style=True, width=float("inf")).strip()


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
        timit_raw = config["datasets"]["timit"]["raw_path"]

    output:
        data_path = directory("outputs/preprocessed_data/timit"),
        notebook_path = "outputs/preprocessing/timit.ipynb"

    shell:
        """
        papermill --log-output notebooks/preprocessing/timit.ipynb \
            {output.notebook_path} \
            -p base_dir {workflow.basedir} \
            -p dataset_path {input.timit_raw} \
            -p out_path {output.data_path}
        """


rule extract_hidden_states:
    input:
        "outputs/preprocessed_data/{dataset}"

    output:
        "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl"

    run:
        outdir = Path(output[0]).parent

        shell("""
        export PYTHONPATH=`pwd`
        python scripts/extract_hidden_states.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            dataset.processed_data_dir={input}
        """)


rule prepare_equivalence_dataset:
    input:
        timit_data = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl"

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


rule run:
    output:
        full_trace = directory("outputs/models/{model_name}/{equivalence_classer}")

    shell:
        """
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            model={wildcards.model_name} \
            equivalence={wildcards.equivalence_classer} \
        """


# Run train without actually training -- used to generate random model weights
rule run_no_train:
    output:
        full_trace = directory("outputs/models/random{model_name}/random")

    shell:
        """
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            model=random{wildcards.model_name} \
            equivalence=phoneme \
            trainer.do_train=false
        """


MODEL_SPEC_LIST = [f"{m['model']}/{m['equivalence']}" for m in config["models"]]
rule run_all:
    input:
        expand("outputs/models/{model_spec}", model_spec=MODEL_SPEC_LIST)


rule run_notebook:
    input:
        notebook = "notebooks/{notebook}.ipynb",
        model_dir = "outputs/models/{model_name}/{equivalence_classer}"
    output:
        outdir = directory("outputs/notebooks/{model_name}/{equivalence_classer}/{notebook}"),
        notebook = "outputs/notebooks/{model_name}/{equivalence_classer}/{notebook}/{notebook}.ipynb"

    shell:
        """
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -p model_dir {input.model_dir} \
            -p output_dir {output.outdir}
        """


rule run_all_notebooks:
    input:
        expand("outputs/notebooks/{model_spec}/{notebook}/{notebook}.ipynb",
                model_spec=MODEL_SPEC_LIST, notebook=ALL_MODEL_NOTEBOOKS)


rule estimate_encoder:
    output:
        model_dir = directory("outputs/encoders/{feature_sets}/{subject}"),
        electrodes = "outputs/encoders/{feature_sets}/{subject}/electrodes.csv",
        scores = "outputs/encoders/{feature_sets}/{subject}/scores.csv",
        predictions = "outputs/encoders/{feature_sets}/{subject}/predictions.npy",
        coefs = "outputs/encoders/{feature_sets}/{subject}/coefs.csv"

    run:
        try:
            data_spec = next(iter(data_spec for data_spec in config["encoding"]["data"] if data_spec["subject"] == wildcards.subject))
        except StopIteration:
            raise ValueError(f"Subject {wildcards.subject} not found in config")

        data_spec = [{"subject": wildcards.subject,
                      "block": block} for block in data_spec["blocks"]]
        data_spec = hydra_param(data_spec)

        shell("""
        python estimate_encoder.py \
            hydra.run.dir={output.model_dir} \
            feature_sets={wildcards.feature_sets} \
            +data='{data_spec}'
        """)


rule compare_encoder_within_subject:
    input:
        model1_output = "outputs/encoders/{comparison_model1}/{subject}/scores.csv",
        model2_output = "outputs/encoders/{comparison_model2}/{subject}/scores.csv"

    output:
        notebook = "outputs/encoder_comparison/{subject}/{comparison_model2}-{comparison_model1}.ipynb",
        csv = "outputs/encoder_comparison/{subject}/{comparison_model2}-{comparison_model1}.csv"

    shell:
        """
        papermill --log-output \
            notebooks/encoding/compare_within_subject.ipynb {output.notebook} \
            -p subject {wildcards.subject} \
            -p model1 {wildcards.comparison_model1} \
            -p model2 {wildcards.comparison_model2} \
            -p output_csv {output.csv}
        """


rule compare_all_encoders_within_subject:
    input:
        lambda _: [f"outputs/encoder_comparison/{subject}/{comp['model2']}-{comp['model1']}.csv"
                  for comp in config["encoding"]["model_comparisons"] for subject in ALL_ENCODING_SUBJECTS]
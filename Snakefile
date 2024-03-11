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


def select_gpu_device(wildcards, resources):
    if resources.gpu == 0:
        return None
    import GPUtil
    available_l = GPUtil.getAvailable(order = 'random', limit = resources.gpu, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
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
rule run_no_train:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = "conf/equivalence/phoneme.yaml",
        model_config = "conf/model/random{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl",
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/phoneme/equivalence.pkl"

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
            equivalence=phoneme \
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
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    resources:
        gpu = 1

    output:
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/embeddings.npy"

    run:
        outdir = Path(embeddings).parent
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python scripts/extract_model_embeddings.py \
            hydra.run.dir={outdir} \
            model={wildcards.model_name} \
            +model.output_dir={input.model_dir} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={wildcards.equivalence_classer} \
            +equivalence.path={input.equivalence_dataset}
        """)



rule run_notebook:
    input:
        notebook = "notebooks/{notebook}.ipynb",
        model_dir = "outputs/models/{model_name}/{equivalence_classer}",

        dataset = "outputs/preprocessed_data/{dataset}",
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl",
        hidden_states = "outputs/hidden_states/{dataset}/{base_model_name}/hidden_states.pkl",
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/embeddings.npy"

    output:
        outdir = directory("outputs/notebooks/{model_name}/{equivalence_classer}/{notebook}"),
        notebook = "outputs/notebooks/{model_name}/{equivalence_classer}/{notebook}/{notebook}.ipynb"

    shell:
        """
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -p model_dir {input.model_dir} \
            -p output_dir {output.outdir} \
            -p dataset_path {input.dataset} \
            -p equivalence_path {input.equivalence_dataset} \
            -p hidden_states_path {input.hidden_states} \
            -p embeddings_path {input.embeddings}
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
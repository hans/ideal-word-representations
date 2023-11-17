
from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()


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


rule run:
    output:
        full_trace = directory("outputs/models/{model_name}"),
        test_result = directory("outputs/models/{model_name}/test_result")

    shell:
        """
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            model={wildcards.model_name}
        """


rule run_notebook:
    input:
        notebook = "notebooks/{notebook}.ipynb",
        model_result = "outputs/models/{model_name}/test_result"
    output:
        outdir = directory("outputs/notebooks/{model_name}/{notebook}"),
        notebook = "outputs/notebooks/{model_name}/{notebook}/{notebook}.ipynb"

    shell:
        """
        papermill --log-output \
            {input.notebook} {output.notebook} \
            -p model_name {wildcards.model_name} \
            -p test_dataset_path {input.model_result}
        """
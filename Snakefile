



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
        "outputs/notebooks/{model_name}/{notebook}.ipynb"

    shell:
        """
        papermill --log-output \
            {input.notebook} {output} \
            -p model_name {wildcards.model_name} \
            -p test_dataset_path {input.model_result}
        """
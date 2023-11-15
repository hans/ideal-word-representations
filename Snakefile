



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
datasets = [
    [
        dict(
            abbr='ARC-c_0',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textA}', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textB}', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textC}', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textD}', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='ARC-Challenge',
            path='opencompass/ai2_arc-dev',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'textA',
                    'textB',
                    'textC',
                    'textD',
                ],
                output_column='answerKey',
                test_range='[0:74]'),
            type='opencompass.datasets.ARCDataset'),
    ],
]
models = [
    dict(
        abbr='entropy_sample_sparsevlm_5e-5_hf-vllm',
        batch_size=8,
        generation_kwargs=dict(stop_token_ids=None),
        max_out_len=256,
        max_seq_len=None,
        model_kwargs=dict(
            gpu_memory_utilization=0.96,
            max_model_len=None,
            max_num_batched_tokens=131072,
            max_num_seqs=1024,
            tensor_parallel_size=1),
        path=
        '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/mistral-7b/data_ratio_12.5/token_ratio_50/entropy_sample_sparsevlm_5e-5',
        run_cfg=dict(num_gpus=1),
        type='opencompass.models.vllm.VLLM'),
]
work_dir = '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/mistral-7b/data_ratio_12.5/token_ratio_50/entropy_sample_sparsevlm_5e-5/results/ARC_c_ppl/20250913_213045'

datasets = [
    [
        dict(
            abbr='ARC-e_1',
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
            name='ARC-Easy',
            path='opencompass/ai2_arc-easy-dev',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'textA',
                    'textB',
                    'textC',
                    'textD',
                ],
                output_column='answerKey',
                test_range='[142:284]'),
            type='opencompass.datasets.ARCDataset'),
    ],
]
models = [
    dict(
        abbr='wise_wise_1e-4_wisely_hf-vllm',
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
        '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/llama2-7b/data_ratio_50.0/token_ratio_50/wise_wise_1e-4_wisely',
        run_cfg=dict(num_gpus=1),
        type='opencompass.models.vllm.VLLM'),
]
work_dir = '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/llama2-7b/data_ratio_50.0/token_ratio_50/wise_wise_1e-4_wisely/results/ARC_e_ppl/20250922_140905'

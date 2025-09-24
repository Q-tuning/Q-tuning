datasets = [
    [
        dict(
            abbr='squad2.0_1',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.SQuAD20Evaluator'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    max_out_len=50,
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            '{context}\nAccording to the above passage, answer the following question. If it is impossible to answer according to the passage, answer `impossible to answer`:\nQuestion: {question}',
                            role='HUMAN'),
                        dict(prompt='Answer:', role='BOT'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/SQuAD2.0/dev-v2.0.json',
            reader_cfg=dict(
                input_columns=[
                    'context',
                    'question',
                ],
                output_column='answers',
                test_range='[2969:5938]'),
            type='opencompass.datasets.SQuAD20Dataset'),
    ],
]
models = [
    dict(
        abbr='random_ppl_1e-4_hf-vllm',
        batch_size=8,
        generation_kwargs=dict(stop_token_ids=None),
        max_out_len=256,
        max_seq_len=None,
        model_kwargs=dict(
            gpu_memory_utilization=0.94,
            max_model_len=None,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            tensor_parallel_size=2),
        path=
        '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/llama2-7b/data_ratio_25/token_ratio_50/random_ppl_1e-4',
        run_cfg=dict(num_gpus=2),
        type='opencompass.models.vllm.VLLM'),
]
work_dir = '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/llama2-7b/data_ratio_25/token_ratio_50/random_ppl_1e-4/results/squad20_gen/20250913_212317'

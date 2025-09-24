datasets = [
    [
        dict(
            abbr='triviaqa_3',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.TriviaQAEvaluator'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    max_out_len=50,
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "Answer these questions, your answer should be as simple as possible, start your answer with the prompt 'The answer is '.\nQ: {question}?",
                            role='HUMAN'),
                        dict(prompt='A:', role='BOT'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='opencompass/trivia_qa',
            reader_cfg=dict(
                input_columns=[
                    'question',
                ],
                output_column='answer',
                test_range='[6630:8840]',
                test_split='dev',
                train_split='dev'),
            type='opencompass.datasets.TriviaQADataset'),
    ],
]
models = [
    dict(
        abbr='entropy_random_1e-4_hf-vllm',
        batch_size=8,
        generation_kwargs=dict(stop_token_ids=None),
        max_out_len=384,
        max_seq_len=None,
        model_kwargs=dict(
            gpu_memory_utilization=0.94,
            max_model_len=None,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            tensor_parallel_size=2),
        path=
        '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/mistral-7b/data_ratio_100/token_ratio_100/entropy_random_1e-4',
        run_cfg=dict(num_gpus=2),
        type='opencompass.models.vllm.VLLM'),
]
work_dir = '/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning/checkpoints/wizard/mistral-7b/data_ratio_100/token_ratio_100/entropy_random_1e-4/results/triviaqa_gen/20250913_162532'

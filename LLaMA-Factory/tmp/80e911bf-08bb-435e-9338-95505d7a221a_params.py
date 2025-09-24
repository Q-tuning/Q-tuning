datasets = [
    [
        dict(
            abbr='race-middle_0',
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
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: A', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: B', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: C', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: D', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='middle',
            path='opencompass/race',
            reader_cfg=dict(
                input_columns=[
                    'article',
                    'question',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='answer',
                test_range='[0:359]',
                test_split='test',
                train_split='validation'),
            type='opencompass.datasets.RaceDataset'),
        dict(
            abbr='race-high_0',
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
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: A', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: B', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: C', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt=
                                'Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}',
                                role='HUMAN'),
                            dict(prompt='A: D', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='high',
            path='opencompass/race',
            reader_cfg=dict(
                input_columns=[
                    'article',
                    'question',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='answer',
                test_range='[0:875]',
                test_split='test',
                train_split='validation'),
            type='opencompass.datasets.RaceDataset'),
    ],
]
models = [
    dict(
        abbr='infobatch_rho_5e-5_hf-vllm',
        batch_size=8,
        generation_kwargs=dict(stop_token_ids=None),
        max_out_len=256,
        max_seq_len=None,
        model_kwargs=dict(
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            max_model_len=8192,
            max_num_seqs=1,
            tensor_parallel_size=1),
        path=
        '/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/checkpoints/alpaca/qw2.5-7b/data_ratio=100/token_ratio=100/infobatch_rho_5e-5',
        run_cfg=dict(num_gpus=1),
        type='opencompass.models.vllm.VLLM'),
]
work_dir = '/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/checkpoints/alpaca/qw2.5-7b/data_ratio=100/token_ratio=100/infobatch_rho_5e-5/results/race_ppl/20250827_145908'

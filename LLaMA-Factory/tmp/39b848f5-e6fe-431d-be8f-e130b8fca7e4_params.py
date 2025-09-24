datasets = [
    [
        dict(
            abbr='race-middle',
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
                test_split='test',
                train_split='validation'),
            type='opencompass.datasets.RaceDataset'),
        dict(
            abbr='race-high',
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
                test_split='test',
                train_split='validation'),
            type='opencompass.datasets.RaceDataset'),
    ],
]
models = [
    dict(
        abbr='random_random_5e-5_hf',
        batch_size=8,
        generation_kwargs=dict(),
        max_out_len=256,
        max_seq_len=None,
        model_kwargs=dict(),
        pad_token_id=None,
        path=
        '/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/checkpoints/alpaca/qw2.5-7b/data_ratio=10/token_ratio=10/random_random_5e-5',
        peft_kwargs=dict(),
        peft_path=None,
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        tokenizer_kwargs=dict(),
        tokenizer_path=None,
        type='opencompass.models.huggingface_above_v4_33.HuggingFaceBaseModel'
    ),
]
work_dir = '/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/checkpoints/alpaca/qw2.5-7b/data_ratio=10/token_ratio=10/random_random_5e-5/results/race_ppl/20250824_143408'

---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:160
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- pearson
- spearman
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-correlation
      name: Cross Encoder Correlation
    dataset:
      name: val evaluator
      type: val_evaluator
    metrics:
    - type: pearson
      value: 0.8561063034315164
      name: Pearson
    - type: spearman
      value: 0.794032386330642
      name: Spearman
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['**Junior Flutter Developer**\n\nWe are seeking a highly motivated Junior Flutter Developer to join our innovative team. This role is perfect for a final-year student or recent graduate passionate about building cross-platform mobile and web applications. You will contribute to developing new features and maintaining existing apps using Flutter, Dart, and Firebase. A solid understanding of OOP, MVC/MVVM, and database integration (Firebase, SQLite) is essential, along with strong problem-solving skills and a collaborative spirit.', 'ID: CV_73.json\nRole: Game Developer\nSkills: Unity, C#, Excel, Word, Powerpoint\nExp: 0 years.'],
    ['We are seeking a highly motivated Junior AI/ML Developer to join our innovative research and development team. This entry-level role is ideal for a recent Computer Science graduate or final-year student passionate about Machine Learning, Natural Language Processing, and Computer Vision. You will contribute to cutting-edge AI solutions, applying strong Python skills and foundational knowledge in Transformers, data preprocessing, and model fine-tuning. This is an excellent opportunity to gain hands-on experience, learn from experts, and make a tangible impact in a dynamic, AI-driven environment.', 'ID: CV_61.json\nRole: Full-stack Developer, Game Developer\nSkills: Software Engineering, Node.js, Express.js, React.js, MongoDB, Docker, HTML/CSS, JavaScript, Java, C#, Unity, SQL Server, MySQL, RESTful API Design, Data Structures and Algorithms\nExp: 0 years.'],
    ['**Junior Full-stack Developer**\n\nWe are seeking a driven Junior Full-stack Developer with a passion for building robust web applications. The ideal candidate possesses hands-on project experience with Node.js and Express.js for backend development, alongside React.js for intuitive frontends. Proficiency in MongoDB, Docker containerization, and RESTful API design is essential. You will contribute to our projects, leveraging your skills in software engineering and modern web technologies. Knowledge of MVC patterns and responsive design is a plus.', 'ID: CV_12.json\nRole: Backend Developer, WordPress PHP Developer\nSkills: PHP, WordPress, MySQL, SEO, Ajax, Scrum Agile, ACF Pro, WPBakery, Elementor, JavaScript, HTML5, CSS3, FileZilla, GitLab, Yoast SEO\nExp: 0 years.'],
    ['**Junior Machine Learning Engineer**\n\nWe are seeking a highly motivated Junior Machine Learning Engineer to contribute to our innovative AI initiatives. This role focuses on developing, training, and optimizing deep learning models for tasks in Natural Language Processing and Computer Vision. You will leverage frameworks like PyTorch and TensorFlow, with opportunities to work with generative AI, transformer architectures, and advanced text/image processing. We value strong Python skills, a collaborative mindset, and practical project experience in AI/ML. Perfect for a recent Computer Science graduate passionate about applied deep learning.', 'ID: CV_66.json\nRole: Server, Barista\nSkills: Network Administration, Network Security, Configuration, Troubleshooting, VPN, Network Design, System Administration, Linux, Windows, Virtualization, Python, SQL, HTML/CSS, Java, Command Line Interface\nExp: 0 years.'],
    ["**Junior AI/Data Engineer**\n\nWe are seeking a highly driven Junior AI/Data Engineer to contribute to our advanced analytics and intelligent systems development. This role involves designing and implementing machine learning algorithms, developing data processing pipelines with big data technologies like Hadoop/Spark, and tackling challenges in areas such as computer vision. You'll leverage strong Python skills, experience with deep learning frameworks (PyTorch/TensorFlow), and a passion for problem-solving. If you thrive on applying theoretical knowledge to real-world data and AI problems, this is your opportunity.", 'ID: CV_1.json\nRole: Software Engineer, Backend Developer, Team Leader\nSkills: Flutter, Python, Node.js, Java, MongoDB, MySQL, Firebase, SQL Server, RESTful APIs, WebSocket, Microservices, MVC, Git, Agile/Scrum, Problem-solving\nExp: 2 years.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    '**Junior Flutter Developer**\n\nWe are seeking a highly motivated Junior Flutter Developer to join our innovative team. This role is perfect for a final-year student or recent graduate passionate about building cross-platform mobile and web applications. You will contribute to developing new features and maintaining existing apps using Flutter, Dart, and Firebase. A solid understanding of OOP, MVC/MVVM, and database integration (Firebase, SQLite) is essential, along with strong problem-solving skills and a collaborative spirit.',
    [
        'ID: CV_73.json\nRole: Game Developer\nSkills: Unity, C#, Excel, Word, Powerpoint\nExp: 0 years.',
        'ID: CV_61.json\nRole: Full-stack Developer, Game Developer\nSkills: Software Engineering, Node.js, Express.js, React.js, MongoDB, Docker, HTML/CSS, JavaScript, Java, C#, Unity, SQL Server, MySQL, RESTful API Design, Data Structures and Algorithms\nExp: 0 years.',
        'ID: CV_12.json\nRole: Backend Developer, WordPress PHP Developer\nSkills: PHP, WordPress, MySQL, SEO, Ajax, Scrum Agile, ACF Pro, WPBakery, Elementor, JavaScript, HTML5, CSS3, FileZilla, GitLab, Yoast SEO\nExp: 0 years.',
        'ID: CV_66.json\nRole: Server, Barista\nSkills: Network Administration, Network Security, Configuration, Troubleshooting, VPN, Network Design, System Administration, Linux, Windows, Virtualization, Python, SQL, HTML/CSS, Java, Command Line Interface\nExp: 0 years.',
        'ID: CV_1.json\nRole: Software Engineer, Backend Developer, Team Leader\nSkills: Flutter, Python, Node.js, Java, MongoDB, MySQL, Firebase, SQL Server, RESTful APIs, WebSocket, Microservices, MVC, Git, Agile/Scrum, Problem-solving\nExp: 2 years.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Correlation

* Dataset: `val_evaluator`
* Evaluated with [<code>CECorrelationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CECorrelationEvaluator)

| Metric       | Value     |
|:-------------|:----------|
| pearson      | 0.8561    |
| **spearman** | **0.794** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 160 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 160 samples:
  |         | sentence_0                                                                                        | sentence_1                                                                                      | label                                                          |
  |:--------|:--------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                            | string                                                                                          | float                                                          |
  | details | <ul><li>min: 467 characters</li><li>mean: 573.39 characters</li><li>max: 693 characters</li></ul> | <ul><li>min: 92 characters</li><li>mean: 218.7 characters</li><li>max: 320 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.38</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | sentence_1                                                                                                                                                                                                                                                                               | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>**Junior Flutter Developer**<br><br>We are seeking a highly motivated Junior Flutter Developer to join our innovative team. This role is perfect for a final-year student or recent graduate passionate about building cross-platform mobile and web applications. You will contribute to developing new features and maintaining existing apps using Flutter, Dart, and Firebase. A solid understanding of OOP, MVC/MVVM, and database integration (Firebase, SQLite) is essential, along with strong problem-solving skills and a collaborative spirit.</code>                                                                | <code>ID: CV_73.json<br>Role: Game Developer<br>Skills: Unity, C#, Excel, Word, Powerpoint<br>Exp: 0 years.</code>                                                                                                                                                                       | <code>0.1</code> |
  | <code>We are seeking a highly motivated Junior AI/ML Developer to join our innovative research and development team. This entry-level role is ideal for a recent Computer Science graduate or final-year student passionate about Machine Learning, Natural Language Processing, and Computer Vision. You will contribute to cutting-edge AI solutions, applying strong Python skills and foundational knowledge in Transformers, data preprocessing, and model fine-tuning. This is an excellent opportunity to gain hands-on experience, learn from experts, and make a tangible impact in a dynamic, AI-driven environment.</code> | <code>ID: CV_61.json<br>Role: Full-stack Developer, Game Developer<br>Skills: Software Engineering, Node.js, Express.js, React.js, MongoDB, Docker, HTML/CSS, JavaScript, Java, C#, Unity, SQL Server, MySQL, RESTful API Design, Data Structures and Algorithms<br>Exp: 0 years.</code> | <code>0.0</code> |
  | <code>**Junior Full-stack Developer**<br><br>We are seeking a driven Junior Full-stack Developer with a passion for building robust web applications. The ideal candidate possesses hands-on project experience with Node.js and Express.js for backend development, alongside React.js for intuitive frontends. Proficiency in MongoDB, Docker containerization, and RESTful API design is essential. You will contribute to our projects, leveraging your skills in software engineering and modern web technologies. Knowledge of MVC patterns and responsive design is a plus.</code>                                             | <code>ID: CV_12.json<br>Role: Backend Developer, WordPress PHP Developer<br>Skills: PHP, WordPress, MySQL, SEO, Ajax, Scrum Agile, ACF Pro, WPBakery, Elementor, JavaScript, HTML5, CSS3, FileZilla, GitLab, Yoast SEO<br>Exp: 0 years.</code>                                           | <code>0.1</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 5

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | val_evaluator_spearman |
|:-----:|:----:|:----------------------:|
| 1.0   | 20   | 0.3333                 |
| 2.0   | 40   | 0.6902                 |
| 3.0   | 60   | 0.7608                 |
| 4.0   | 80   | 0.7865                 |
| 5.0   | 100  | 0.7940                 |


### Framework Versions
- Python: 3.13.5
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.0+cpu
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
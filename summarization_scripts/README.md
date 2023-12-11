# Zero-shot and few-shot multilingual abstractive summarization
## Master thesis - Vladimir Solovyev
## Institute: [Artificial Intelligence for Language Technologies (AI4LT)](https://ai4lt.anthropomatik.kit.edu/english/index.php)

### Installation:
- python version 3.10
- Multilingual rouge scorer
```bash
git clone https://github.com/csebuetnlp/xl-sum.git
cd xl-sum/multilingual_rouge_scoring
pip install -r requirements.txt
pip install ./
```
- PyTorch
```bash
pip install torch==1.13.1 torchvision torchaudio
```
- Fairseq project
```bash
git clone https://github.com/vladsolovyev/fairseq_summarization.git
cd fairseq_summarization
pip install -e ./
```
- Evaluate
```bash
pip install evaluate==0.4.0
```
- Nltk
```bash
pip install nltk==3.8.1
```
- Langid
```bash
pip install langid==1.1.6
```
- SentencePiece
```bash
pip install sentencepiece
```
- Transformers
```bash
pip install transformers==4.29.2
```
- GPUtil
```bash
pip install GPUtil==1.4.0
```
- BERTScore
```bash
pip install bert-score==0.3.9
```

### Datasets:
1. [XL-Sum](https://github.com/csebuetnlp/xl-sum)
   - Multilingual abstractive summarization. Multilingual monolingual article-summary pairs from BBC in 44 languages.
   - In my thesis, I utilize four languages: English, Spanish, Russian, and Gujarati. Gujarati is a special case since it is considered a low-resource language both in the xl-sum dataset and in the pretraining of [mBART](https://arxiv.org/pdf/2001.08210.pdf)
   - For training, I exclude certain samples that have very short articles (consisting of no more than 20 sentence parts) or very short summaries (consisting of no more than 10 sentence parts).
   - Original statistics and benchmarks are available on the [xl-sum GitHub webpage](https://github.com/csebuetnlp/xl-sum)
   - The statistics after removing some samples from the training subset can be found in the file [stat_xlsum.txt](../summarization_datasets/stat_xlsum.txt)
   - [Paper describing the dataset](https://aclanthology.org/2021.findings-acl.413.pdf)

2. [WikiLingua](https://github.com/esdurmus/Wikilingua)
   - article and summary pairs from [WikiHow](https://www.wikihow.com/Main-Page).
   - Multilingual monolingual and cross-lingual data in 18 languages.
   - For my experiments, I utilize monolingual data in English, Spanish, and Russian, as well as cross-lingual data pairs such as Spanish-English, Russian-English, and Turkish-English. The evaluation is conducted using the same cross-lingual pairs (Spanish-English, Russian-English, and Turkish-English) as used in the baseline.
   - The baseline and a detailed description of the dataset can be found in the [paper](https://arxiv.org/pdf/2010.03093.pdf).
   - The statistics of the dataset can be found in the file: [stat_wikilingua.txt](../summarization_datasets/stat_wikilingua.txt)

### Data and model preprocessing scripts
- mBART model
```bash
bash ../summarization_datasets/load_mbart_model.sh
```
- XL-Sum
```bash
python ../summarization_datasets/prepare_data_xlsum.py
python preprocess_data_xlsum.py
```
- WikiLingua
```bash
python ../summarization_datasets/prepare_data_wikilingua.py
python preprocess_data_wikilingua.py
```
- Data for two-step fine-tuning
```bash
python ../summarization_datasets/prepare_data_nmt.py
python preprocess_data_nmt.py
```
### Experiments

#### Running experiments
```bash
bash cuda_training_runner.sh
```
- Scripts for running experiments
  - [Script for all experiments](./training_runner.py)
  - [Script for intralingual experiments](./training_runner_xlsum.py)
  - [Script for cross-lingual experiments](./training_runner_wikilingua.py)
  - [Script for training a single model](./train_summarization.py)
  - [Script for summaries generation using a single model](./generate_summaries.py)

### Metrics for evaluation
- Rouge - [paper](https://aclanthology.org/W04-1013.pdf); [description with code examples](https://huggingface.co/spaces/evaluate-metric/rouge)
- BERTScore - [paper](https://arxiv.org/pdf/1904.09675.pdf); [description with code examples](https://huggingface.co/spaces/evaluate-metric/bertscore)
- Langid - [paper](https://aclanthology.org/P12-3005/); [examples](https://github.com/saffsd/langid.py)

#### Intralingual using XL-Sum dataset

How to create monolingual summaries of text in low-resource languages (zero-shot and few-shot), given only monolingual data in other languages and a pretrained mBART model?

Two types of models are trained: 1) using data only in English 2) using data in English, Spanish, and Russian jointly.
Evaluation is conducted using data in Spanish, Russian, and Gujarati.

Results description:
1. *_supervised - model is trained using a combination of monolingual English, Spanish, Russian, and Gujarati data jointly.
2. en_XX - model trained and evaluated using only english data.
3. *_zero - model trained using only English data as in the experiment no. 2; zero-shot experiments are conducted using test data in Spanish, Russian, and Gujarati
4. *_translated - translate test data in Spanish, Russian, and Gujarati into English, create summaries in English using a model trained with English data from the experiment no. 2, translate summaries back into Spanish, Russian, and Gujarati, and calculate metrics.
5. *_10/100/1000/10000 - use a model trained with English data from the experiment no. 2 and conduct few-shot experiments. Tune the model using 10/100/1000/10000 samples in Spanish, Russian, and Gujarati.
6. *_multiEnEsRu - train a multilingual intralingual model using English, Spanish, and Russian data jointly. Evaluate using test datasets of all 4 languages (also Gujarati, another zero-shot experiment for Gujarati).
7. gu_IN_multiEnEsRu_10/100/1000 - tune multilingual model from the experiment no. 7 using 10/100/1000 samples from gujarati train data and evaluate using gujarati test data

Configurations:
1. "base_model" - baseline and few-shot results ([results](./xlsum_results/2023-10-12/base_model/metrics.csv)
2. "unfrozen_embeddings" - model with fine-tuned embeddings ([results](./xlsum_results/2023-10-12/unfrozen_embeddings/metrics.csv)
3. "frozen_decoder" - only encoder layers are fine-tuned ([results](./xlsum_results/2023-10-12/frozen_decoder/metrics.csv)
4. "frozen_except_attn_and_layer_norm" - only self-attention in encoder layers and encoder-attention in decoder layers and normalization layers are fine-tuned ([results](./xlsum_results/2023-10-12/frozen_except_attn_and_layer_norm/metrics.csv)
5. "frozen_except_attn_qk" - only queries and keys in self-attention in encoder layers and queries and keys in encoder-attention in decoder layers are fine-tuned ([results](./xlsum_results/2023-10-12/frozen_except_attn_qk/metrics.csv)

#### Cross-lingual using WikiLingua dataset

How to create cross-lingual summaries of text (zero-shot and few-shot), given only monolingual data in various languages and a pretrained mBART model? Additionally, test a scenario when input and/or output languages are unseen during training.

Training is conducted using monolingual data in English, Spanish, and Russian.
Evaluation is performed for the following language pairs:
- Spanish-English
- Russian-English
- Turkish-English
- Spanish-Russian
- English-Turkish
- Turkish-Turkish

Results description:
1. *_supervised - model trained using cross-lingual data for all language pairs jointly
2. *_mono - a multilingual model trained using monolingual training data in three languages (English, Spanish, and Russian) jointly. Zero-shot evaluation for all pairs.
3. *_mono_10/100/1000/10000 - few shot experiments. Tune multilingual model from the experiment no. 2 using 10/100/1000/10000 samples
4. *_mono_adv_nll - experiments using an original adversarial loss classifier
5. *_mono_adv_kldivloss - experiments using a newly proposed adversarial classifier
6. *_mono_adv_nll_tuned - use the model from the experiment no. 4, freeze encoder and fine-tune decoder for 30000 steps  
7. *_mono_adv_kldivloss_tuned - use the model from the experiment no. 5, freeze encoder and fine-tune decoder for 30000 steps
8. *_translated - translation-based solution

Configurations:
1. "base_model_with_adv" - baselines, few-shot results, adversarial loss, adversarial loss with KL-divergence ([results](./wiki_results/2023-10-12/base_model_with_adv/metrics.csv)
2. "residual_drop_at_7" - experiments with removed residual connection in the 7th encoder layer ([results](./wiki_results/2023-10-12/residual_drop_at_7/metrics.csv)
3. "encoder_output" - experiments with language-specific encoder output adapters ([results](./wiki_results/2023-10-12/encoder_output/metrics.csv)
4. "decoder_adapter_tgt_lang_id" - experiments with language-specific layer adapters ([results](./wiki_results/2023-10-12/decoder_adapter_tgt_lang_id/metrics.csv)

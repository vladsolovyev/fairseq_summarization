# Zero-shot and few-shot multilingual abstractive summarization
## Master thesis - Vladimir Solovyev
## Institute: [Artificial Intelligence for Language Technologies (AI4LT)](https://ai4lt.anthropomatik.kit.edu/english/index.php)

### Datasets:
1. [xl-sum](https://github.com/csebuetnlp/xl-sum)
   - Multilingual abstractive summarization. Multilingual monolingual article-summary pairs from BBC in 44 languages.
   - I use 4 languages for my thesis: English, Spanish, Russian and Burmese. Burmese is a special case as it is a low-resource as well in the xl-sum dataset as in the pretraining of [mbart](https://arxiv.org/pdf/2001.08210.pdf)
   - For training, I remove some samples which have very short articles (not more than 20 sentence parts) or very short summaries (not more than 10 sentence parts).
   - Original statistic and baselines are provided on the [xl-sum github webpage](https://github.com/csebuetnlp/xl-sum)
   - Statistic after removing some samples from the training subset: [stat_xlsum.txt](../summarization_datasets/stat_xlsum.txt)
   - [Script for parsing](../summarization_datasets/prepare_data_xlsum.py)
   - [Paper describing the dataset](https://aclanthology.org/2021.findings-acl.413.pdf)
2. [Wikilingua](https://github.com/esdurmus/Wikilingua)
   - article and summary pairs from [WikiHow](https://www.wikihow.com/Main-Page).
   - Multilingual monolingual and multilingual cross-lingual data in 18 languages.
   - I use monolingual (english, spanish and russian) and cross-lingual (spanish-english and russian-english) data for experiments. Evaluation is done using cross-lingual (spanish-english and russian-english) as in the baseline.
   - Baseline and description of the dataset can be found in the [paper](https://arxiv.org/pdf/2010.03093.pdf).
   - Statistic of the dataset: [stat_wikilingua.txt](../summarization_datasets/stat_wikilingua.txt)
   - [Script for parsing](../summarization_datasets/prepare_data_wikilingua.py)

### Experiments

#### xl-sum
1. en_XX, es_XX, ru_RU, my_MM - monolingual models, which are trained and evaluated for every language separately.
2. es_XX_zero, ru_RU_zero, my_MM_zero - use monolingual english model from the experiment no. 1 and perform zero-shot experiments using test data of spanish, russian and burmese
3. es_XX_translated, ru_RU_translated, my_MM_translated - translate test data of spanish, russian and burmese into english, create summaries in english using monolingual english model from the experiment no. 1, translate summaries back into spanish, russian and burmese, and evaluate using test datasets of these languages.
4. es_XX_tuned_10/100/1000/10000, ru_RU_tuned_10/100/1000/10000, my_MM_tuned_10/100/1000 - use monolingual english model from the experiment no. 1 and perform few-shot experiments. Tune english model using 10/100/1000/10000 samples from train dataset of spanish, russian and burmese. Evaluate using test datasets of these languages.
5. es_XX_tuned_all, ru_RU_tuned_all, my_MM_tuned_all - use monolingual english model from the experiment no. 1 and tune it using complete train datasets of spanish, russian and burmese separately. Evaluate using test datasets of these languages.
6. en_XX_multilingual, es_XX_multilingual, ru_RU_multilingual, my_MM_multilingual - train one multilingual model using only english, spanish and russian together. Evaluate using test datasets of all 4 languages (also burmese, another few-shot experiment for burmese) separately.
7. my_MM_multilingual_tuned_burmese - tune multilingual model from the experiment no. 6 using burmese train data and evaluate using burmese test data

- TODO: add evaluation using [flan-ul2](https://huggingface.co/google/flan-ul2)
- [Experiments runner](./training_runner_xlsum.py)
- First results (without some experiments and should be rerun because of new parameters in the training setup):
  1. [Without embeddings freezing](./2023-03-06/metrics_without_freezing.csv)
  2. [With embeddings freezing](./2023-03-07/metrics_with_freezing.csv)

#### Wikilingua
1. es_XX-en_XX, ru_RU-en_XX - two cross-lingual cases separately
2. es_XX-en_XX_zero, ru_RU-en_XX_zero - firstly a multilingual model is trained using monolingual training data of three languages(english, spanish and russian). After that, two cross-lingual cases are evaluated separately using test data without any fine-tuning.
3. es_XX-en_XX_tuned_10/100/1000/10000, ru_RU-en_XX_tuned_10/100/1000/10000 - few shot experiments. Tune multilingual model from the experiment no. 2 using few data from spanish-english and russian-english datasets
4. es_XX-en_XX_tuned_all, ru_RU-en_XX_tuned_all - the same as no. 3 but fine-tuning two models separately with complete training data of spanish-english and russian-english datasets
5. es_XX-en_XX_tuned_together, ru_RU-en_XX_tuned_together - the same as no.4 but fine-tuning only one model using complete training data of both spanish-english and russian-english datasets together

- TODO: add evaluation using [flan-ul2](https://huggingface.co/google/flan-ul2)
- [Experiments runner](./training_runner_wikilingua.py)

- First results :
  1. [Without embeddings freezing](./2023-03-27_wiki/metrics_without_freezing.csv)
  2. With embeddings freezing - in process

### Experiments setup
- All experiments are run using [fairseq library](https://github.com/facebookresearch/fairseq)
- [Script for training](./train_summarization.py)
- [Script for summaries generation](./generate_summaries.py)

### TODO
- Add residual connections removing.
- Run all experiments in 4 variants:
  - without embeddings freezing and without residual connections removing
  - without embeddings freezing but with residual connections removing
  - with embeddings freezing but without residual connections removing
  - with embeddings freezing and with residual connections removing
- Discuss other variants if applicable

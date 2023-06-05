# Zero-shot and few-shot multilingual abstractive summarization
## Master thesis - Vladimir Solovyev
## Institute: [Artificial Intelligence for Language Technologies (AI4LT)](https://ai4lt.anthropomatik.kit.edu/english/index.php)

### Datasets:
1. [xl-sum](https://github.com/csebuetnlp/xl-sum)
   - Multilingual abstractive summarization. Multilingual monolingual article-summary pairs from BBC in 44 languages.
   - I use 4 languages for my thesis: English, Spanish, Russian and Gujarati. Gujarati is a special case as it is a low-resource as well in the xl-sum dataset as in the pretraining of [mbart](https://arxiv.org/pdf/2001.08210.pdf)
   - For training, I remove some samples which have very short articles (not more than 20 sentence parts) or very short summaries (not more than 10 sentence parts).
   - Original statistic and baselines are provided on the [xl-sum github webpage](https://github.com/csebuetnlp/xl-sum)
   - Statistic after removing some samples from the training subset: [stat_xlsum.txt](../summarization_datasets/stat_xlsum.txt)
   - [Script for parsing](../summarization_datasets/prepare_data_xlsum.py)
   - [Paper describing the dataset](https://aclanthology.org/2021.findings-acl.413.pdf)
2. [Wikilingua](https://github.com/esdurmus/Wikilingua)
   - article and summary pairs from [WikiHow](https://www.wikihow.com/Main-Page).
   - Multilingual monolingual and multilingual cross-lingual data in 18 languages.
   - I use monolingual (english, spanish and russian) and cross-lingual (spanish-english, russian-english and turkish-english) data for experiments. Evaluation is done using cross-lingual (spanish-english, russian-english and turkish-english) as in the baseline.
   - Baseline and description of the dataset can be found in the [paper](https://arxiv.org/pdf/2010.03093.pdf).
   - Statistic of the dataset: [stat_wikilingua.txt](../summarization_datasets/stat_wikilingua.txt)
   - [Script for parsing](../summarization_datasets/prepare_data_wikilingua.py)

### Experiments

#### xl-sum
1. en_XX_baseline, es_XX_baseline, ru_RU_baseline, gu_IN_baseline - baseline model trained using monolingual english, spanish, russian and gujarati data together.
2. en_XX - monolingual model trained and evaluated using only english data.
3. es_XX_zero, ru_RU_zero, gu_IN_zero - use monolingual english model from the experiment no. 2 and perform zero-shot experiments using test data of spanish, russian and gujarati
4. es_XX_translated, ru_RU_translated, gu_IN_translated - translate test data of spanish, russian and gujarati into english, create summaries in english using monolingual english model from the experiment no. 2, translate summaries back into spanish, russian and gujarati, and evaluate using test datasets of these languages.
5. es_XX_tuned_10/100/1000/10000, ru_RU_tuned_10/100/1000/10000, gu_IN_tuned_10/100/1000 - use monolingual english model from the experiment no. 2 and perform few-shot experiments. Tune english model using 10/100/1000/10000 samples from train dataset of spanish, russian and gujarati. Evaluate using test datasets of these languages.
6. es_XX_tuned_all, ru_RU_tuned_all, gu_IN_tuned_all - use monolingual english model from the experiment no. 2 and tune it using complete train datasets of spanish, russian and gujarati separately. Evaluate using test datasets of these languages.
7. en_XX_multiEnEsRu, es_XX_multiEnEsRu, ru_RU_multiEnEsRu, gu_IN_multiEnEsRu - train one multilingual model using only english, spanish and russian together. Evaluate using test datasets of all 4 languages (also gujarati, another few-shot experiment for gujarati) separately.
8. gu_IN_multiEnEsRu_10/100/1000 - tune multilingual model from the experiment no. 7 using 10/100/1000 samples from gujarati train data and evaluate using gujarati test data
9. gu_IN_multiEnEsRu_all - tune multilingual model from the experiment no. 7 using complete gujarati train data and evaluate using gujarati test data
10. en_XX_multiEnEsRu_adv, es_XX_multiEnEsRu_adv, ru_RU_multiEnEsRu_adv, gu_IN_multiEnEsRu_adv - tune multilingual model from the experiment 7 using adversarial loss and english, spanish and russian data together. Evaluate using test datasets of all 4 languages (also gujarati, another few-shot experiment for gujarati) separately.
11. gu_IN_multiEnEsRu_adv_10/100/1000 - tune multilingual model from the experiment no. 10 using 10/100/1000 samples from gujarati train data and evaluate using gujarati test data
12. gu_IN_multiEnEsRu_adv_all - tune multilingual model from the experiment no. 10 using complete gujarati train data and evaluate using gujarati test data

- TODO: add evaluation using [flan-ul2](https://huggingface.co/google/flan-ul2)
- [Experiments runner](./training_runner_xlsum.py)
- TODO: update all results as experiments have been changed
- All experiments are run in 4 possible configurations:
  - one of three options: residual connections drop (4th layer); parameters freezing of the first 6 layers, none of both

#### Wikilingua
1. es_XX_baseline, ru_RU_baseline, tr_TR_baseline - three crosslingual cases (spanish-english, russian-english and turkish-english) trained separately as a baseline
2. es_XX_mono, ru_RU_mono, tr_TR_mono - firstly a multilingual model is trained using monolingual training data of three languages(english, spanish and russian). After that, three cross-lingual cases (spanish-english, russian-english and turkish-english) are evaluated separately using test data without any fine-tuning.
3. es_XX_mono_10/100/1000/10000, ru_RU_mono_10/100/1000/10000, tr_TR_mono_10/100/1000 - few shot experiments. Tune multilingual model from the experiment no. 2 using few data from spanish-english, russian-english and turkish-english datasets
4. es_XX_mono_all, ru_RU_mono_all, tr_TR_mono_all - the same as no. 3 but fine-tuning three models separately with complete training data of spanish-english, russian-english and turkish-english datasets
5. es_XX_mono_adv, ru_RU_mono_adv, tr_TR_mono_adv - finetune a multilingual model from the experiment no. 2 using adversarial loss and monolingual data of three languages(english, spanish and russian). Evaluate using spanish-english, russian-english and turkish-english datasets
6. es_XX_mono_adv_10/100/1000/10000, ru_RU_mono_adv_10/100/1000/10000, tr_TR_mono_adv_10/100/1000 - few shot experiments. Tune multilingual model from the experiment no. 5 using few data from spanish-english, russian-english and turkish-english datasets
7. es_XX_mono_adv_all, ru_RU_mono_adv_all, tr_TR_mono_adv_all - the same as no. 6 but fine-tuning three models separately with complete training data of spanish-english, russian-english and turkish-english datasets


- TODO: add evaluation using [flan-ul2](https://huggingface.co/google/flan-ul2)
- [Experiments runner](./training_runner_wikilingua.py)
- TODO: update all results as experiments have been changed

### Experiments setup
- All experiments are run using [fairseq library](https://github.com/facebookresearch/fairseq)
- [Script for training](./train_summarization.py)
- [Script for summaries generation](./generate_summaries.py)
- All experiments are run in 6 possible configurations:
  - one of three options: residual connections drop (4th layer); parameters freezing of the first 6 layers, none of both
  - and one of two other options: with or without use of language embeddings for an encoder output

### Metrics for evaluation
- Rouge - [paper](https://aclanthology.org/W04-1013.pdf); [description with code examples](https://huggingface.co/spaces/evaluate-metric/rouge)
- BertScore - [paper](https://arxiv.org/pdf/1904.09675.pdf); [description with code examples](https://huggingface.co/spaces/evaluate-metric/bertscore)

### Configuration explanation
- An idea of residual connections drop is described in the [paper](https://aclanthology.org/2021.acl-long.101.pdf). An example is [here](https://github.com/dannigt/fairseq/tree/master/examples/residual_drop)
- Parameters freezing of the first 6 layers allows the use of transfer learning from the pretrained mBart model and not to over-fit to specific language.
- Adversarial loss implementation leads to better language-independent representations and better zero-shot cross-lingual summarization. An idea is taken from the [first](https://arxiv.org/pdf/2211.01292.pdf) and the [second](https://arxiv.org/pdf/1903.07091.pdf) papers. An implementation is partly copied and adopted from [here](https://github.com/dannigt/fairseq/tree/master/examples/adapter_transformer)
- Language embeddings for an encoder output can help to output an expected language. I just add embeddings to encoder outputs.
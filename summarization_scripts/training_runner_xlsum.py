import shutil
from datetime import datetime

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU", "my_MM"]
lenpen = "0.6"
rouge_scorer = "multilingual"


def run_xlsum_experiments(freeze_embeddings=False, encoder_drop_residual=None, prefix=""):
    metrics = dict()
    output_dir = "xlsum/{}".format(prefix)
    # every language separately
    for language in languages:
        checkpoint_dir = "{}/xlsum/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  save_dir=checkpoint_dir,
                                  freeze_embeddings=freeze_embeddings,
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics[language] = generate_and_evaluate_summaries(directory="xlsum",
                                                            source_language=language,
                                                            target_language=language,
                                                            lang_pairs="{}-{}".format(language, language),
                                                            checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                            lenpen=lenpen,
                                                            rouge_scorer=rouge_scorer)
        if language != "en_XX":
            shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # zero shot. Evaluate spanish, russian and burmese datasets using english model
    for language in languages[1:]:
        metrics["{}_zero".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()

    # input is translated from spanish, russian and burmese into english. Create summaries using english model.
    # Translate summaries in english back into spanish, russian and burmese
    # and evaluate using original data in these languages.
    for translation_language in ["es", "ru", "my"]:
        metrics["{}_translated".format(translation_language)] = \
            generate_and_evaluate_summaries(directory="xlsum_{}_en".format(translation_language),
                                            source_language="en_XX",
                                            target_language="en_XX",
                                            lang_pairs="en_XX-en_XX",
                                            checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                            lenpen=lenpen,
                                            translate_to_lang=translation_language,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments. Tune english model using few data from spanish, russian and burmese datasets
    for language in languages[1:]:
        for data_size in [10, 100, 1000, 10000]:
            if language == "my_MM" and data_size == 10000:
                break
            checkpoint_dir = "{}/xlsum_{}/{}".format(output_dir, data_size, language)
            train_summarization_model(data_dir="xlsum_{}".format(data_size),
                                      lang_pairs="{}-{}".format(language, language),
                                      checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                      save_dir=checkpoint_dir,
                                      max_epoch="5",
                                      freeze_embeddings=freeze_embeddings,
                                      encoder_drop_residual=encoder_drop_residual)
            free_memory()
            metrics["{}_tuned_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="xlsum",
                                                source_language=language,
                                                target_language=language,
                                                lang_pairs="{}-{}".format(language, language),
                                                checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                rouge_scorer=rouge_scorer)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune english model using complete data from spanish, russian and burmese datasets
    for language in languages[1:]:
        checkpoint_dir = "{}/xlsum_all/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  freeze_embeddings=freeze_embeddings,
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics["{}_tuned_all".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # train using english, spanish and russian together and evaluate with all 4 languages
    checkpoint_dir = "{}/multilingual".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:-1]]),
                              save_dir=checkpoint_dir,
                              freeze_embeddings=freeze_embeddings,
                              encoder_drop_residual=encoder_drop_residual)
    free_memory()
    for language in languages:
        metrics["{}_multilingual".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()

    # tune multilingual model and evaluate it using burmese dataset
    checkpoint_dir = "{}/multilingual_tuned_burmese".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs="my_MM-my_MM",
                              checkpoint="{}/multilingual/checkpoint_last.pt".format(output_dir),
                              save_dir=checkpoint_dir,
                              freeze_embeddings=freeze_embeddings,
                              encoder_drop_residual=encoder_drop_residual)
    free_memory()
    metrics["my_MM_multilingual_tuned_burmese"] = \
        generate_and_evaluate_summaries(directory="xlsum",
                                        source_language="my_MM",
                                        target_language="my_MM",
                                        lang_pairs="my_MM-my_MM",
                                        checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                        lenpen=lenpen,
                                        rouge_scorer=rouge_scorer)
    save_metrics(metrics, output_dir)
    free_memory()

    shutil.rmtree(checkpoint_dir)
    shutil.rmtree("{}/xlsum/en_XX".format(output_dir))
    shutil.rmtree("{}/multilingual".format(output_dir))


if __name__ == "__main__":
    run_xlsum_experiments()

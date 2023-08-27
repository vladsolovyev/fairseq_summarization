from summarization_scripts.train_summarization import train_summarization_model

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]


def train_nmt_models():
    language_pairs_4 = list()
    for i in range(len(languages))[:-1]:
        for k in range(len(languages))[i + 1:]:
            language_pairs_4.append("{}-{}".format(languages[i], languages[k]))
            language_pairs_4.append("{}-{}".format(languages[k], languages[i]))
    print(language_pairs_4)
    language_pairs_3 = list()
    for i in range(len(languages))[:-2]:
        for k in range(len(languages))[i + 1:-1]:
            language_pairs_3.append("{}-{}".format(languages[i], languages[k]))
            language_pairs_3.append("{}-{}".format(languages[k], languages[i]))
    print(language_pairs_3)

    for dir_name, language_pairs, freeze_decoder_layers, use_decoder_adapter, use_encoder_output_adapter in zip(
            ["3_langs", "4_langs", "3_langs_frozen_decoder", "4_langs_frozen_decoder",
             "4_langs_decoder_adapter", "4_langs_encoder_output_adapter"],
            [language_pairs_3, language_pairs_4, language_pairs_3, language_pairs_4, language_pairs_4, language_pairs_4],
            [False, False, True, True, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, True]
    ):
        checkpoint_dir = "translated_pretrained/{}".format(dir_name)
        train_summarization_model(data_dir="translated",
                                  lang_pairs=",".join(language_pairs),
                                  save_dir=checkpoint_dir,
                                  freeze_decoder_layers=freeze_decoder_layers,
                                  freeze_elements="attn_and_layer_norm",
                                  use_decoder_adapter=use_decoder_adapter,
                                  use_encoder_output_adapter=use_decoder_adapter)

        train_summarization_model(data_dir="translated",
                                  lang_pairs=",".join(language_pairs),
                                  checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                  save_dir="{}/adv_nll".format(checkpoint_dir),
                                  use_adversarial_loss=True,
                                  max_update="100000",
                                  validate=False,
                                  append_src_tok=False,
                                  use_kldivloss=False,
                                  freeze_decoder_layers=freeze_decoder_layers,
                                  freeze_elements="attn_and_layer_norm",
                                  use_decoder_adapter=use_decoder_adapter,
                                  use_encoder_output_adapter=use_decoder_adapter)

        train_summarization_model(data_dir="translated",
                                  lang_pairs=",".join(language_pairs),
                                  checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                  save_dir="{}/adv_kldivloss".format(checkpoint_dir),
                                  use_adversarial_loss=True,
                                  max_update="100000",
                                  validate=False,
                                  append_src_tok=False,
                                  use_kldivloss=True,
                                  freeze_decoder_layers=freeze_decoder_layers,
                                  freeze_elements="attn_and_layer_norm",
                                  use_decoder_adapter=use_decoder_adapter,
                                  use_encoder_output_adapter=use_decoder_adapter)


if __name__ == "__main__":
    train_nmt_models()

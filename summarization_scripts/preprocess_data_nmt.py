import numpy as np
from datasets import load_dataset

from summarization_scripts.utils import preprocess_data

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]


def main():
    for i in range(len(languages))[:-1]:
        for k in range(len(languages))[i + 1:]:
            preprocess_data(languages[i],
                            languages[k],
                            "translated/{}_{}".format(languages[i], languages[k]),
                            "translated",
                            add_validation_data=True)
            preprocess_data(languages[k],
                            languages[i],
                            "translated/{}_{}".format(languages[i], languages[k]),
                            "translated",
                            add_validation_data=True)


if __name__ == "__main__":
    main()

DICT=../summarization_datasets/mbart.cc25.v2/dict.txt
DATA=../summarization_datasets/xlsum/parser
fairseq-preprocess --source-lang input_text.en --target-lang summary.en \
    --trainpref $DATA/train --validpref $DATA/valid --testpref $DATA/test \
    --destdir xlsum_data \
	--srcdict $DICT \
    --tgtdict $DICT \
    --workers 20

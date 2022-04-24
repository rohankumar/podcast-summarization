vocab_size=200000
position_markers=0
export LC_ALL=C
cat train.document train.summary |
  tr -s '[:space:]' '\n' |
  sort |
  uniq -c |
  sort -k1,1bnr -k2 |
  head -n "$((vocab_size - 4))" |
  awk '{ print $2 " " $1 }' >dict.pg.txt
python3 -c "[print('<unk-{}> 0'.format(n)) for n in range($position_markers)]" >>dict.pg.txt

./preprocess.py --source train.document --target train.summary --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out train.pg.src --target-out train.pg.tgt
./preprocess.py --source dev.document --target dev.summary --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out valid.pg.src --target-out valid.pg.tgt
./preprocess.py --source test.document --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out test.pg.src

fairseq-preprocess \
  --source-lang src \
  --target-lang tgt \
  --trainpref train.pg \
  --validpref valid.pg \
  --destdir bin \
  --workers 60 \
  --srcdict dict.pg.txt \
  --joined-dictionary


total_updates=10000
warmup_updates=500
lr=0.001
max_tokens=4096
update_freq=4
pointer_layer=-2

CUDA_VISIBLE_DEVICES=0 fairseq-train bin \
    --user-dir pointer_generator_src \
    --max-tokens "$max_tokens" \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --required-batch-size-multiple 1 \
    --arch transformer_pointer_generator \
    --alignment-layer "$pointer_layer" \
    --alignment-heads 1 \
    --source-position-markers 1000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr "$lr" --max-update "$total_updates" --warmup-updates "$warmup_updates" \
    --update-freq "$update_freq" \
    --skip-invalid-size-inputs-valid-test

batch_size=32
beam_size=6
max_length=500
length_penalty=1.0

fairseq-interactive bin \
    --user-dir pointer_generator_src \
    --batch-size "$batch_size" \
    --task translation \
    --source-lang src --target-lang tgt \
    --path checkpoints/checkpoint_last.pt \
    --input test.pg.src \
    --buffer-size 200 \
    --max-len-a 0 \
    --max-len-b "$max_length" \
    --lenpen "$length_penalty" \
    --beam "$beam_size" \
    --skip-invalid-size-inputs-valid-test |
    tee generate.out
grep ^H generate.out | cut -f 3- >generate.hyp

./postprocess.py \
    --source <(awk 'NF<1024' test.document) \
    --target generate.hyp \
    --target-out generate.hyp.processed
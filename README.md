# sdg_multi_label_classifiaction

## Two-stage SBERT fine-tuning

```
python multi_label.py \
--num_training 128 \
--seed 42 \
--num_iter 5 \
--multi_label_finetuning \
--label_desc_finetuning
```

## Single-stage label description fine-tuning

```
python multi_label.py \
--num_training 128 \
--seed 42 \
--num_iter 5 \
--label_desc_finetuning
```

## Single-stage OOD dataset fine-tuning

```
python multi_label.py \
--num_training 128 \
--seed 42 \
--num_iter 5 \
--multi_label_finetuning 
```

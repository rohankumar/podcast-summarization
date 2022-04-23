#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import LEDForConditionalGeneration, LEDTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_metric, load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# In[2]:


checkpoint = 'allenai/led-base-16384'

tokenizer = LEDTokenizerFast.from_pretrained(checkpoint)

model = LEDForConditionalGeneration.from_pretrained(checkpoint, gradient_checkpointing=True, use_cache=False)
model.config.num_beams = 4
model.config.max_length = 512
model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3


# In[3]:


rouge = load_metric("rouge")


# In[4]:




# In[6]:

podcast_train = load_from_disk('data/processed/brass_train.dataset')
podcast_dev = load_from_disk('data/processed/brass_dev.dataset')

batch_size = 2

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
#     fp16_backend="apex",
    output_dir="./",
    logging_steps=250,
    eval_steps=5000,
    save_steps=500,
    warmup_steps=1500,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
)


# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }



trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=podcast_train['train'],
    eval_dataset=podcast_dev['train'], 
)

trainer.train()


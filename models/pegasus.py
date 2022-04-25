#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd

from tqdm import tqdm
from pynvml import *
from transformers import Trainer

from datasets import load_dataset, load_metric
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import wandb


# In[2]:


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"
configuration = PegasusConfig()
wandb.init()


# In[17]:


dataset_train = load_dataset('csv', data_files = "../data/cleaned/train_clean.csv")


# In[5]:


dataset_val = load_dataset('csv', data_files = "../data/cleaned/dev_clean.csv")


# In[4]:


model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)


# In[18]:


max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['transcript'], padding = "longest" , truncation = True)
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['episode_description'], padding="longest", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[19]:


train_inp = dataset_train.map(preprocess_function, batched=True)


# In[21]:


dev_inp = dataset_val.map(preprocess_function, batched=True)


# In[22]:


print_gpu_utilization()


# In[5]:


model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


# In[24]:


print_gpu_utilization()


# In[25]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[26]:


import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


# In[32]:


training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    weight_decay=0.01,
    num_train_epochs=10,
    optim = "adafactor",
    run_name="pegasus_first_itr",
    eval_accumulation_steps = 500,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_inp['train'],
    eval_dataset=dev_inp['train'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()


# In[9]:


# read test dataset
# generate predictions

df = pd.read_csv("../data/cleaned/test_clean.csv")
df.head()


# In[ ]:


f = open("generated.txt", "w")
for idx, row in df.iterrows():
    src_text = row['transcript']
    batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    f.write(str(tgt_text[0]))
    f.write("\n")


# In[ ]:





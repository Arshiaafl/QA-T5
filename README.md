### `README.md`


# QA-T5: Question Answering with T5

This repository contains code and resources for fine-tuning a T5 model for the task of question rewriting on the Disfluent QA dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)


## Project Overview

The QA-T5 project focuses on using the T5 transformer model to rewrite disfluent questions into fluent questions. This can be particularly useful in improving the quality of user-generated questions in various NLP applications.

## Features

- Fine-tune a T5 model for question rewriting.
- Support for loading datasets from both local JSON files and the Hugging Face Hub.
- Evaluation of model performance using ROUGE, BELU, and METEOR metrics.
- Efficient dataset preprocessing with Hugging Face `datasets`.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Arshiaafl/QA-T5.git
   cd QA-T5
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv qa
   source qa/bin/activate  # On Windows use `.\qa\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Load Dataset

You can load the dataset from the Hugging Face Hub or from local JSON files.

**Load from the Hugging Face Hub:**

```python
def hub_dataset():
    """
    Loads a dataset from the Hugging Face Hub and split it into training and validation sets.

    Args:
        

    Returns:
        tuple: A tuple containing the dataset
    """
    # Loads the dataset
    dataset = load_dataset("google-research-datasets/disfl_qa", split="validation")
    
    
    return dataset
```

**Load from JSON Files:**

```python
def json_dataset(json_file_path):
    """
    Load a dataset from a JSON file and split it into training and validation sets.

    Args:
        json_file_path (str): Path to the JSON file containing the dataset.

    Returns:
        tuple: A tuple containing the dataset
    """
    # Loads and process data from JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    data = {
        "disfluent question": [],
        "original question": []
    }

    for key, value in json_data.items():
        data["disfluent question"].append(value['disfluent'])
        data["original question"].append(value['original'])

    # Converts data to Hugging Face Dataset format
    dataset = Dataset.from_dict(data)
    return dataset
```

### Training the Model

To train the model, you can use the `Seq2SeqTrainer` from the Hugging Face `transformers` library. Make sure to set up the training arguments and datasets correctly.

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```
### Loading the Model

Alternatively, you can also load the fine-tuned model using this code:
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="my_fine_tuned_t5_small_model", min_length=5, max_length=42, num_beams=5, length_penalty=0.4)
```
### Results

After training, the model's performance can be evaluated using the ROUGE score or any other relevant metric. The results will be stored in the specified output directory.



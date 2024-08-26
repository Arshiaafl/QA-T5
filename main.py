from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, ProgressCallback
import pandas as pd
from datasets import load_dataset, Dataset, load_metric
import json



model_name = "t5-small"  
tokenizer = T5Tokenizer.from_pretrained(model_name)


def hub_dataset(validation_split=0.1):
    """
    Loads a dataset from the Hugging Face Hub and splits it into training and validation sets.

    Args:
        validation_split (float): Fraction of the training dataset to use as validation.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    # Loads the dataset
    full_dataset = load_dataset("google-research-datasets/disfl_qa", split="train")
    
    # Converts the full dataset to a Pandas DataFrame
    df = full_dataset.to_pandas()
    
    # Splits the dataset into train and validation sets
    train_df, val_df = train_test_split(df, test_size=validation_split, random_state=42)
    
    # Converts the DataFrames back to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def json_dataset(json_file_path, validation_split=0.1):
    """
    Loads a dataset from a JSON file and splits it into training and validation sets.

    Args:
        json_file_path (str): Path to the JSON file containing the dataset.
        validation_split (float): Fraction of the training dataset to use as validation.

    Returns:
        tuple: A tuple containing the training and validation datasets.
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
    full_dataset = Dataset.from_dict(data)
    
    # Converts the full dataset to a Pandas DataFrame for splitting
    df = full_dataset.to_pandas()
    
    # Splits the DataFrame into train and validation sets
    train_df, val_df = train_test_split(df, test_size=validation_split, random_state=42)
    
    # Converts the split DataFrames back to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    
    return train_dataset, val_dataset


def preprocess_function(examples):
    # Prepends the string "summarize: " to each document in the 'text' field of the input examples.
    # This is done to instruct the T5 model on the task it needs to perform, which in this case is summarization.
    inputs = ["summarize: " + doc for doc in examples["disfluent question"]]

    # Tokenizes the prepended input texts to convert them into a format that can be fed into the T5 model.
    # Sets a maximum token length of 1024, and truncates any text longer than this limit.
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Tokenizes the 'summary' field of the input examples to prepare the target labels for the summarization task.
    # Sets a maximum token length of 128, and truncates any text longer than this limit.
    labels = tokenizer(text_target=examples["original question"], max_length=128, truncation=True)

    # Assigns the tokenized labels to the 'labels' field of model_inputs.
    # The 'labels' field is used during training to calculate the loss and guide model learning.
    model_inputs["labels"] = labels["input_ids"]

    # Returns the prepared inputs and labels as a single dictionary, ready for training.
    return model_inputs


train_dataset, val_dataset = json_dataset("train.json")

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

print(tokenized_train_dataset)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")


rouge = evaluate.load("rouge")



def compute_metrics(eval_pred):
    # Unpacks the evaluation predictions tuple into predictions and labels.
    predictions, labels = eval_pred

    # Decodes the tokenized predictions back to text, skipping any special tokens (e.g., padding tokens).
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replaces any -100 values in labels with the tokenizer's pad_token_id.
    # This is done because -100 is often used to ignore certain tokens when calculating the loss during training.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decodes the tokenized labels back to text, skipping any special tokens (e.g., padding tokens).
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Computes the ROUGE metric between the decoded predictions and decoded labels.
    # The use_stemmer parameter enables stemming, which reduces words to their root form before comparison.
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Calculates the length of each prediction by counting the non-padding tokens.
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    # Computes the mean length of the predictions and adds it to the result dictionary under the key "gen_len".
    result["gen_len"] = np.mean(prediction_lens)

    # Rounds each value in the result dictionary to 4 decimal places for cleaner output, and returns the result.
    return {k: round(v, 4) for k, v in result.items()}



model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")


training_args = Seq2SeqTrainingArguments(
    output_dir="my_fine_tuned_t5_small_model_cp",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs",  # Directory to save logs
    logging_steps=10,  # Log every 10 steps
)

# Initialize Trainer with the above training arguments
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Starts training
h = trainer.train()

# Saves the model
trainer.save_model("my_fine_tuned_t5_small_model")
print(h)

# Converts the training log history to a DataFrame
log_history = trainer.state.log_history
df_logs = pd.DataFrame(log_history)

df_logs.to_csv("logs_df.csv", sep='\t', encoding='utf-8', index=False, header=True)

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import os

#Sets the directory where your tf event files are stored
log_dir = "./logs"  # Replace with the directory containing your event files

# Initializes the event accumulator
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

# Extract the scalar data for training and validation loss
train_loss = event_acc.Scalars('train/loss')  # Replace 'loss' with the actual name used in your logs
val_loss = event_acc.Scalars('eval/loss')  # Replace 'eval_loss' with the actual name used in your logs

# Converts the scalar data to a pandas DataFrame
df_train_loss = pd.DataFrame(train_loss)
df_val_loss = pd.DataFrame(val_loss)

plt.figure(figsize=(12, 6))

# Plots training loss over steps with a smooth line
plt.plot(df_train_loss['step'], df_train_loss['value'], label='Training Loss', color='blue', linestyle='-', linewidth=1.5)

# Plots validation loss (once per epoch) with distinct markers
plt.plot(df_val_loss['step'], df_val_loss['value'], label='Validation Loss', color='orange', linestyle='--', marker='o', markersize=8)

# Annotates the epochs on the validation loss points
for i, txt in enumerate(range(1, len(df_val_loss) + 1)):
    plt.annotate(f'Epoch {txt}', 
                 (df_val_loss['step'][i], df_val_loss['value'][i]), 
                 textcoords="offset points", 
                 xytext=(5,5), 
                 ha='center', fontsize=10, color='orange')

# Adding axis labels and a title
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training vs Validation Loss Across Epochs", fontsize=14)

# Adding grid for better readability
plt.grid(visible=True, linestyle='--', linewidth=0.5)

# Adding a legend to differentiate between the plots
plt.legend(loc='upper right', fontsize=12)

# Indicates epoch points on x-axis with vertical dashed lines
for i in range(len(df_val_loss)):
    plt.axvline(x=df_val_loss['step'][i], color='grey', linestyle='--', linewidth=0.8)

# Displays the plot
plt.show()
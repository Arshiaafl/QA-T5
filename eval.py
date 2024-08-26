from transformers import pipeline

text = "summarize: " + " What is, no sorry. Who is Donald Trump?"

summarizer = pipeline("summarization", model="my_fine_tuned_t5_small_model", min_length=5, max_length=42, num_beams=5, length_penalty=0.4)
# The summarizer returns a list of dictionaries. Get the first dictionary and extract the 'summary_text' value.
pred = summarizer(text)

# Extract the summary text
summary = pred[0]['summary_text']

print(summary)

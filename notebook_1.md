# Working with Hugging Face

## Getting Started with Hugging Face

### Searching the Hub with Python

The Hugging Face Hub provides a nice user interface for searching for
models and learning more about them. At times, you may find it
convenient to be able to do the same thing without leaving the
development environment. Fortunately, Hugging Face also provides a
Python package which allows you to find models through code.

Use the `huggingface_hub` package to find the most downloaded model for
text classification.

`HfApi` and `ModelFilter` from the `huggingface_hub` package is already
loaded for you.

**Instructions**

- Create the instance of the API to the Hugging Face Hub.
- Return a list of one item which is the most downloaded model text
  classification task.
- Store the returned object as a list named `modelList`.

**Answer**

```{python}

```

### Saving a model

There may be situations where downloading and storing a model locally
(i.e. a directory on your computer) is desired. For example, if working
offline.

Practice downloading and saving here. An instance of `AutoModel` is
already loaded for you under the same name.

**Instructions**

- Instantiate the model class for the
  `distilbert-base-uncased-finetuned-sst-2-english` model.
- Save the model as the `modelId` under "models/".

**Answer**

```{python}

```

### Inspecting datasets

The datasets on Hugging Face range in terms of size, information, and
features. Therefore it's beneficial to inspect it before committing to
loading a dataset into your environment.

Let's inspect the "derenrich/wikidata-en-descriptions-small" dataset.

*Note: this exercise may take a minute due to the dataset size.*

**Instructions**

- Import `load_dataset_builder`.
- Create the dataset builder to inspect the dataset.
- Print the features for the dataset.

**Answer**

```{python}

```

### Loading datasets

Hugging Face built the `dataset` package for interacting with datasets.
There are a lot of convenient functions, including
`load_dataset_builder` which we just used. After inspecting a dataset to
ensure its the right one for your project, it's time to load the
dataset! For this, we can leverage input parameters for `load_dataset`
to specify which parts of a dataset to load, i.e. the "train" dataset
for English wikipedia articles.

The `load_dataset` module from the `datasets` package is already loaded
for you. Note: the `load_dataset` function was modified for the purpose
of this exercise.

**Instructions**

- Load the `"wikimedia/wikipedia"` dataset and save as `wikipedia`.

**Answer**

```{python}

```

### Manipulating datasets

There will likely be many occasions when you will need to manipulate a
dataset before using it within a ML task. Two common manipulations are
filtering and selecting (or slicing). Given the size of these datasets,
Hugging Face leverages arrow file types.

This means performing manipulations are slightly different than what you
might be used to. Fortunately, there's already methods to help with
this!

The dataset is already loaded for you under `wikipedia`.

**Instructions**

- Filter the dataset for rows with the term "football" in the `text`
  column and save as `filtered`.
- Select a single example from the filtered dataset and save as
  `example`.

**Answer**

```{python}

```

## Building Pipelines with Hugging Face

### Getting started with pipelines

Hugging Face has an ecosystem of libraries that allow users to leverage
tools at different levels. The `pipeline` module from the `transformers`
library is a great place to get started with performing ML tasks. It
removes the requirement for training models, allowing for quicker
experimentation and results. It does this by being a wrapper around
underlying objects, functions, and processes.

Getting started with pipeline can be done by defining a task or model.
This helps with quick experimentation as you become familiar with the
library.

Create your first pipelines for sentiment analysis. The `input` is a
sentence string that is already loaded for you.

**Instructions**

- Import `pipeline` from the `transformers` library.
- Create the first pipeline by specifying the task "sentiment-analysis"
  and save as `task_pipeline`.
- Create another pipeline but only specify the model,
  `distilbert-base-uncased-finetuned-sst-2-english` and save as
  `model_pipeline`.
- Predict the sentiment of `input` using both pipelines.

**Answer**

```{python}

```

### Using AutoClasses

AutoClasses offer more control for machine learning tasks, and they can
also be used with `pipeline()` for quick application. It's a nice
balance of control and convenience.

Continue with the sentiment analysis task and combine AutoClasses with
the pipeline module.

`AutoModelForSequenceClassification` and `AutoTokenizer` from the
`transformers` library have already been imported for you and the input
text is saved as `input`.

**Instructions**

- Download the model and tokenizer for
  `"distilbert-base-uncased-finetuned-sst-2-english"` and save as
  `model` and `tokenizer`, respectively.
- Create the pipeline using `model` and `tokenizer` and save as
  `sentimentAnalysis`.
- Predict the output using `sentimentAnalysis` and save as `output`.

**Answer**

```{python}

```

### Comparing models with the pipeline

One of the great benefits of the `pipeline()` module is the ease at
which you can experiment with different models simply by changing the
`"model"` input. This is a good way to determine which model works best
for a particular task or dataset that you are working with.

Experiment with two sentiment analysis models by creating pipelines for
each, then using them to predict the sentiment for a sentence.

`pipeline` from the `transformers` library is already loaded for you.
The example input sentence is saved as `input`.

**Instructions**

- Create a pipeline for labeling text as positive or negative, using the
  model `"distilbert-base-uncased-finetuned-sst-2-english"`, and save as
  `distil_pipeline`.
- Predict the sentiment for the `input` and save as `distil_output`.

**Answer**

```{python}

```

### Normalizing text

An important step to performing an NLP task is tokenizing the input
text. This makes the text more understandable and manageable for the ML
models, or other algorithms.

Before performing tokenization, it's best to run normalization steps,
i.e. removing white spaces and accents, lowercasing, and more. Each
tokenizer available in Hugging Face uses it's own normalization and
tokenization processes.

Let's take a look at what normalization the `distilbert-base-uncased`
tokenizer applies to the `input_string`, "HOWDY, how aré yoü?".

**Instructions**

- Import `AutoTokenizer` from `transformers`.
- Download the tokenizer for `distilbert-base-uncased` using
  `AutoTokenizer` and save as `tokenizer`.
- Run the normalization process for the `tokenizer`.

**Answer**

```{python}

```

### Comparing tokenizer output

Most models in Hugging Face will have an associated tokenizer that will
help prepare the input data based on what the model expects. After
normalization, the tokenizer will split the input into smaller chunks
based on the chosen algorithm. This is known as "pre-tokenization".

Let's explore the different types of pre-tokenization by performing this
process with two tokenizers on the same input. We will be using
`DistilBertTokenizer` and `GPT2Tokenizer` which have already been loaded
for you. The input text string, "Pineapple on pizza is pretty good, I
guess" is saved as `input`.

**Instructions**

- Download the `"gpt2"` tokenizer and save as `gpt_tokenizer`.
- Use `gpt_tokenizer` to create the tokens from the `input`.
- Repeat the same two steps for
  `"distilbert-base-uncased-finetuned-sst-2-english"`.

**Answer**

```{python}

```

### Grammatical correctness

Text classification is the process of labeling an input text into a
pre-defined category. This can take the form of sentiment - `positive`
or `negative` - spam detection - `spam` or `not spam` - and even
grammatical errors.

Explore the use of a `text-classification` pipeline for checking an
input sentence for grammatical errors.

`pipeline` from the `transformers` library is already loaded for you.

**Instructions**

- Create a pipeline for the task `text-classification` and use the model
  `"abdulmatinomotoso/English_Grammar_Checker"`, saving the pipeline as
  `classifier`.
- Use the `classifier` to predict the grammatical correctness of the
  input sentence provided and save as `output`.

**Answer**

```{python}

```

### Question Natural Language Inference

Another task under the text classification umbrella is Question Natural
Language Inference, or QNLI. This determines if a piece of text contains
enough information to answer a posed question. This requires the model
to perform logical reasoning which are important for Q&A applications.

Performing different tasks with the `text-classification` pipeline can
be done by choosing different models. Each model is trained to predict
specific labels and optimized for learning different context within a
text.

`pipeline` from the `transformers` library is already loaded for you.

**Instructions**

- Create a text classification QNLI pipeline using the model
  `"cross-encoder/qnli-electra-base"` and save as `classifier`.
- Use this classifier to predict if the text contains information to
  answer the question.

**Answer**

```{python}

```

### Zero-shot classification

Zero-shot classification is the ability for a transformer to predict a
label from a new set of classes which it wasn't originally trained to
identify. This is possible through its transfer learning capabilities.
It can be an extremely valuable tool.

Hugging Face `pipeline()` also has a `zero-shot-classification` task.
These pipelines require both an input text and candidate labels.

Build a zero-shot classifier to predict the label for the input `text`,
a news headline that has been loaded for you.

`pipelines` from the `transformers` library is already loaded for you.
Note that we are using our own version of the pipeline function to
enable you to learn how to use these functions without having to
download the model.

**Instructions**

- Build the pipeline for a zero-shot-classification task and save as
  `classifier`.
- Create a list of the labels - "politics", "science", "sports" - and
  save as `candidate_labels`.
- Predict the label of `text` using the classifier and candidate labels.

**Answer**

```{python}

```

### Summarizing long text

Summarization is a useful task for reducing large piece of text into
something more manageable. This could be beneficial for multiple reasons
like reducing the amount of time a reader needs to spend to obtain the
important point of a piece of text.

The Hugging Face `pipeline()` task, "summarization", builds a s
summarization pipeline which is a quick way to perform summarization on
a piece of text. You'll do that by creating a new pipeline and using it
to summarize a piece of text from a Wikipedia page on Greece.

`pipeline` from the `transformers` library and the `original_text` have
already been loaded for you.

**Instructions**

- Create the summarization `pipeline` using the task "summarization" and
  save as `summarizer`.
- Use the new pipeline to create a summary of the text and save as
  `summary_text`.
- Compare the length of the original and summary text.

**Answer**

```{python}

```

### Using min_length and max_length

The `pipeline()` function, has two important parameters: `min_length`
and `max_length`. These are useful for adjusting the length of the
resulting summary text to be short, longer, or within a certain number
of words. You might want to do this if there are space constraints
(i.e., small storage), to enhance readability, or improve the quality of
the summary.

You'll experiment with a short and long summarizer by setting these two
parameters to a small range, then a wider range.

`pipeline` from the `transformers` library and the `original_text` have
already been loaded for you.

**Instructions**

- Create a summarization pipeline using a `min_length` of 1 and
  `max_length` of 10; save as `short_summarizer`.
- Summarize the `original_text` using the `short_summarizer` and save
  the result as `short_summary_text`.

**Answer**

```{python}

```

### Summarizing several inputs

Often times, you'll be working on projects where summarization will
occur over an entire dataset or list of items, not just a single piece
of text. Fortunately, this can be done by passing in a list of text
items. This will return a list of summarized texts.

You'll build a final summarization pipeline and use it to summarize a
list of text items from the `wiki` dataset.

`pipeline` from the `transformers` library and the dataset `wiki` have
already been loaded for you.

**Instructions**

- Create a list of text items to summarize from the `wiki` dataset and
  save as `text_to_summarize`.
- Create a summarization pipeline using a `min_length` of 20 and a
  `max_length` of 50 and save as `summarizer`.
- Summarize the first three items in the `text_to_summarize` list
  setting `truncation` to `True`.
- Create a for-loop that will print each text summary.

**Answer**

```{python}

```

## Building Pipelines for Image and Audio

### Processing image data

Just like text inputs, image inputs will typically require
pre-processing before using with a pipeline for an image-based machine
learning task, such as image classification. Some common transformations
include cropping and resizing. Fortunately, Hugging Face provides
modules for performing these steps via the `image_transforms` module in
the `transformers` library.

Use this module to apply a transformation to a fashion image.

<img
src="https://assets.datacamp.com/production/repositories/6536/datasets/b8f0d79998622cb33ed09a3c48f1d20b25712bbe/fashion.jpeg"
height="500" />

`image_transforms` from the `transformers` library has already been
loaded for you, as well as the JPEG saved as `original_image`.

**Instructions**

- Convert the image to a numpy array.
- Crop the center of the image to keep a new 200 x 200 image using
  `image_transforms` and save as `cropped_image`.

**Answer**

```{python}

```

### Creating an image classifier

Image classification is the process of labeling an image based on the
content. This is useful for many reasons from improving search to saving
agriculture crops from disease. It is also helpful for identifying
clothing items in fashion photos.

Build an image classification pipeline using a model trained on
identifying clothing items to classify the image you just cropped.

Both `pipeline` from the `transformers` library and the image, saved as
`cropped_image`, have already been loaded for you.

**Instructions**

- Create the image classifier pipeline using the model provided and save
  as `image_classifier`.
- Predict the class of the `cropped_image` and save as `results`.
- Print the predicted `"label"` of the result.

**Answer**

```{python}

```

### Document question and answering

Document question and answering is a multi-modal ML task which analyzes
an image of a document, such as a contract, converts it to text, and
allows a question to be asked about the text. This is useful when there
are many scanned documents which need to be searched, for example
financial records.

Build a pipeline for document question and answering, then ask the
pre-loaded question `Which meeting is this document about?`.

`pipeline` from the `transformers` library and the `question` are
already loaded for you. Note that we are using our own pipeline and dqa
functions to enable you to learn how to use these functions without some
of the extra setup. Please visit the Hugging Face documentation to dive
deeper.

**Instructions**

- Create a pipeline for `document-question-answering` and save as `dqa`.
- Save the path to the image, `document.png`, as `image`.
- Get the answer for the `question` of the `image` using the `dqa`
  pipeline and save as `results`.

**Answer**

```{python}

```

### Visual question and answering

Visual question and answering is an ML task that attempts to provide the
best answer for a question about an image. The model will analyze the
content of the image and return a label as the answer.

For example, if asking about the clothes a model is wearing, the model
may return clothing items as the label. Such a task can be beneficial
for people who are visually impaired or as a classification method
(similar to image classification but more open-ended).

`pipeline` from the `transformers` library and both the `question` and
`image` are already loaded for you.

**Instructions**

- Create a visual question and answering pipeline by setting the `task`
  to `visual-question-answering` and save as `vqa`.
- Use the `vqa` pipeline to get an answer for the `image` and
  `question`, then save as `results`.

**Answer**

```{python}

```

### Resampling audio files

The sampling rate of an audio file determines the resolution. The higher
the sampling rate, the higher the resolution which provides more detail
about the sound wave itself.

When performing ML tasks it is important to ensure each file has the
same sampling rate. This will maintain consistency and prepare the audio
files based on what the model expects regarding number of data points
per audio file.

The dataset, `audio_file`, and `Audio` from the `datasets` library are
already loaded for you.

**Instructions**

- Save the old sampling rate, for reference, as `old_sampling_rate`.
- Resample the `audio` column to a new rate of 16,000 kHz and save to
  `audio_file`.
- Compare the old and new sampling rates.

**Answer**

```{python}

```

### Filtering out audio files

There will be occasions where you will want, or need, to filter a
dataset based on a specific criteria. A common example of this is
filtering for audio files that are under a specified duration.

The `librosa` and `numpy` libraries, as well as the `dataset`, have
already been loaded for you. Note: we have modified the `librosa`
library for the purposes of this exercise, but the functionality and
pattern is the same.

**Instructions**

- Loop over each `row` of the audio paths in the `dataset` and calculate
  the duration, appending to the `old_durations_list`.
- Create a new column called `duration` using `old_durations_list` and
  save to `dataset`.

**Answer**

```{python}

```

### Classifying audio files

Audio classification can be used for any task that requires labeling a
piece of audio based on its content. A common use case is identifying
spoken languages.

You will do just that using an example from the `common_language`
dataset. The model, `facebook/mms-lid-126` from Meta is a common model
used for this task given its coverage of languages.

`pipeline` from the `transformers` library as well as the `dataset` have
been loaded for you. It has been modified for the purposes of this
exercise.

**Instructions**

- Create the pipeline for audio classification and save as `classifier`.
- Extract the sample audio and sentence and save as `audio` and
  `sentence`, respectively.
- Predict the label for the audio using the `classifier` and save as
  `prediction`.

**Answer**

```{python}

```

### Instantiating an ASR pipeline

You've been tasked with generating text from a dataset of audio files.
Accuracy is important, so you need to make sure you choose the best
model for automatic speech recognition. You also don't have the time to
train your own model.

Compare the predictions between the Wav2Vec and Whisper models by
instantiating two pipelines for automatic speech recognition. You want
to test out the functionality, so you should try it out on one example
first.

`pipeline` from the `transformers` package is already loaded for you.
Likewise, the dataset has already been loaded for you, resampled, and
saved as `english`. One audio file with it's associated metadata is
saved as `example`.

**Instructions**

- Instantiate the first automatic speech recognition pipeline for the
  "wav2vec2" model from Meta.
- Predict the text from the `example` audio.
- Repeat these two steps for the "whisper-tiny" model from OpenAI in
  order to compare the predictions.

**Answer**

```{python}

```

### Word error rate

The Wav2Vec and Whisper models predicted very similar text with only
some minor differences. Luckily, for the first this example record, you
have the true sentence for reference. You can use Word Error Rate (WER)
to determine which model quantitatively performed the best.

`load` from the `evaluate` package has already loaded for you. Likewise,
the `example` and predictions - `meta_pred` and `open_pred` - were saved
from the previous exercise.

**Instructions**

- Instantiate the word error rate metric object and save as `wer`.
- Save the true sentence of the example as `true_sentence`.
- Compute the word error rate for each model prediction and save as
  `meta_wer` and `open_wer`, respectively.

**Answer**

```{python}

```

### Iterating over a dataset

You were able test the functionality and understand the performance on
one example from the dataset. Now, let's evaluate the models over the
first 100 audio files to make a final decision about which is best for
this dataset.

In order to do this efficiently, you can create a function that will
iterate over the rows of the dataset and yield a set of audio and true
sentence pairs on each iteration.

The dataset, `english`, ASR models - `meta_asr` and `open_asr` - and
`pandas` have all been loaded for you.

**Instructions**

- Create the `data()` function to iterate over the first 3 rows of the
  dataset.
- Within `data()`, predict the text for each audio file using both the
  `meta_asr` and `open_asr` pipelines.
- Append the results as a dictionary in the `output` list.

**Answer**

```{python}

```

## Fine-tuning and Embeddings

### Preparing a dataset

Fine-tuning a model requires several steps including identifying the
model to fine-tune, preparing the dataset, creating the training loop
object, then saving the model.

A model trained on English text classification has been identified for
you, but it's up to you to prepare the imdb dataset in order to
fine-tune this model to classify the sentiment of movie reviews.

The imdb dataset is already loaded for you and saved as `dataset`.

**Instructions**

- Import `AutoModelForSequenceClassification` and `AutoTokenizer`.

**Answer**

```{python}

```

### Building the trainer

To fine-tune a model, it must be trained on new data. This is the
process of the model learning patterns within a training dataset, then
evaluating how well it can predict patterns in an unseen test dataset.
The goal is to help the model build an understanding of patterns while
also being generalizable to new data yet to be seen.

Build a training object to fine-tune the
"distilbert-base-uncased-finetuned-sst-2-english" model to be better at
identifying sentiment of movie reviews.

The `training_data` and `testing_data` dataset are available for you.
`Trainer` and `TrainingArguments` from `transformers` are also loaded.
They were modified for the purpose of this exercise.

**Instructions**

- Create the training arguments object setting the output directory to
  `./results`.
- Create the trainer object by passing in the model, training arguments,
  `training_data` and `testing_data`.
- Start the trainer.

**Answer**

```{python}

```

### Using the fine-tuned model

Now that the model is fine-tuned, it can be used within `pipeline`
tasks, such as for sentiment analysis. At this point, the model is
typically saved to a local directory (i.e. on your own computer), so a
local file path is needed.

You'll use the newly fine-tuned distilbert model. There is a sentence,
"I am a HUGE fan of romantic comedies.", saved as `text_example`.

Note: we are using our own `pipeline` module for this exercise for
teaching purposes. The model is "saved" (i.e. not really) under the path
`./fine_tuned_model`.

**Instructions**

- Create the classifier pipeline for `"sentiment-analysis"` and use the
  model saved in `"./fine_tuned_model"`.
- Classify the text, `text_example`, and save the results as `results`.

**Answer**

```{python}

```

### Generating text from a text prompt

Generating text can be accomplished using Auto classes from the Hugging
Face `transformers` library. It can be a useful method for developing
content, such as technical documentation or creative material.

You'll walk through the steps to process the text prompt, "Wear
sunglasses when its sunny because", then generate new text from it.

`AutoTokenizer` and `AutoModelForCausalLM` from the `transformers`
library are already loaded for you.

**Instructions**

- Get the tokenizer and the model for 'gpt2' then save as `tokenizer`
  and `model`, respectively.

**Answer**

```{python}

```

### Generating a caption for an image

Generating text can be done for modalities other than text, such as
images. This has a lot of benefits including faster content creation by
generating captions from images.

You'll create a caption for a fashion image using the Microsoft GIT
model ("microsoft/git-base-coco").

`AutoProcessor` and `AutoModelForCausalLM` from the `transformers`
library is already loaded for you along with the `image`.

**Instructions**

- Get the processor and model and save with the same names,
  respectively.
- Process the `image` using the processor and save as `pixels`.
- Generate the output and save as `output`.
- Decode the `output` and save as `caption`.

**Answer**

```{python}

```

### Generate embeddings for a sentence

Embeddings are playing an increasingly big role in ML and AI systems. A
common use case is embedding text to support search.

The `sentence-transformers` package from Hugging Face is useful for
getting started with embedding models. You'll compare the embedding
shape from two different models - "all-MiniLM-L6-v2" and
"sentence-transformers/paraphrase-albert-small-v2". This can determine
which is better suited for a project (i.e. because of storage
constraints).

The sentence used for embedding, *"Programmers, do you put your comments
(before\|after) the related code?"*, is saved as `sentence`.

`SentenceTransformer` from the `sentence-transformers` package was
already loaded for you.

**Instructions**

- Create the first embedder using `SentenceTransformer` and the model
  `all-MiniLM-L6-v2`, then save as `embedder1`.
- Use `embedder1` to generate an embedding for the `sentence` and save
  as `embedding1`.
- Repeat these two steps for the `paraphrase-albert-small-v2` model.

**Answer**

```{python}

```

### Using semantic search

The similarity, or closeness, between a query and the other sentences,
or documents, is the foundation for semantic search. This is a search
method which takes into account context and intent of the query.
Similarity measures, such as cosine similarity, are used to quantify the
distance between the query and each sentence within the dimensional
space. Results of a search are based on the closest sentences to the
query.

You will use semantic search to return the top two Reddit threads
relevant to the user query, "I need a desktop book reader for Mac".

The `embedder` and `sentence_embeddings` are already loaded for you
along with `util.semantic_search()`.

**Instructions**

- Generate embeddings for the query and save as `query_embedding`.
- Use `util.semantic_search` to compare the query embedding with the
  sentence embeddings and save as `hits`.
- Complete the `for-loop` to print the results from `hits[0]`.

**Answer**

```{python}

```

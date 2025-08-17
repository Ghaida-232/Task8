# English–Arabic Machine Translation with Fine-Tuning

## Objective
- Understand the structure and requirements of machine translation datasets.
- Preprocess bilingual corpora for training.
- Apply fine-tuning on a lightweight LLM-based model (opus-mt-en-ar) for English-toArabic translation.
- Evaluate translation quality using BLEU and example comparisons.
- Deliver an end-to-end Jupyter Notebook with clear documentation.

## Dataset overview

  We used the [Arabic–English Sentence Bank 25k dataset](https://www.kaggle.com/datasets/tahaalselwii/the-arabic-english-sentence-bank-25k/data) from Kaggle which contains 25,000 aligned sentence pairs in both English and Arabic. The dataset is provided in JSON and CSV formats making it easy to load and explore. 
It includes mix of everyday and motivational sentences providing a rich source of general purpose conversational and inspirational language suitable for fine-tuning translation models.


### Dataset preview
<img width="1367" height="309" alt="image" src="https://github.com/user-attachments/assets/addde6fb-30f9-4b7f-a0cc-750d80737809" />
<img width="807" height="412" alt="image" src="https://github.com/user-attachments/assets/1c190c39-aac5-4bb5-b746-a25d3d526558" />
<br/>

- The dataset contains **25,000 aligned English–Arabic sentence pairs** with three main fields:  
  - `English` – the source sentence in English  
  - `Arabic` – the corresponding translation in Arabic  
  - `Type` – sentence category (e.g., *Inspirational Sentences*)  
- **Shape**: (25,000 rows × 3 columns)  
- **Dtypes**: all fields stored as `object` (string)  
- **Missing values**: none detected.


## Exploratory Data Analysis

### Sentence length distribution
<img width="1014" height="470" alt="image" src="https://github.com/user-attachments/assets/363c1ea5-8bbe-4333-b3f6-8bfb982f9115" /> 
<br/>
  Both English and Arabic sentences are mostly short to medium length with English sentences averaging 9.65 words and Arabic 8.35 words. The distribution is similar for both languages with most sentences between 7–11 words. This balance is ideal for training translation models as it reduces alignment issues.

### Vocabulary size
<img width="858" height="404" alt="image" src="https://github.com/user-attachments/assets/07334b0d-85ed-428d-9972-cd0759dd30dd" />
<br/>

 - English: **9,533 unique words**
 - Arabic: **16,185 unique words**  

The larger Arabic vocabulary reflects the morphological richness and diversity of the language.

### Language token balance
<img width="1415" height="410" alt="image" src="https://github.com/user-attachments/assets/a1a8d27d-65fb-497e-8c47-50d3895f608b" />
<br/>

 - Total English tokens: **241,208**
 - Total Arabic tokens: **208,665**
 - Average tokens per sentence: **EN 9.65**, **AR 8.35**

This shows a slight length difference but overall the dataset is well-balanced.
    
### Domain-specific terminology frequency
<img width="2488" height="662" alt="image" src="https://github.com/user-attachments/assets/2780cda7-26ff-43b8-93f6-b890a6d50371" />
<br/>

The Most Frequent 20 Terms 
 - English: the, to, you, i, a, in, is, this, be, if, of, what, will, more, would, your, do, we, life, with
 - Arabic: من, في, أن, أكثر, على, هذه, إلى, هو, لو, الذي, ما, هل, لكانت, إذا, كنت, تم, تحسين, الشخص, لا, أعتقد

This indicates the dataset covers a wide range of everyday and motivational language.

### Sentence Statistics
<img width="810" height="614" alt="image" src="https://github.com/user-attachments/assets/d64200b2-fa81-4c8f-b76b-032492d6399c" />
<br/>

- Minimum sentence length: 2 words (both languages)
- Maximum: 25 (EN), 19 (AR)
- 50% of sentences: 8–10 words (EN), 7–10 words (AR)
- Standard deviation: 2.95 (EN), 2.52 (AR) The dataset is consistent and avoids extreme outliers.

### Alignment Quality
<img width="1562" height="768" alt="image" src="https://github.com/user-attachments/assets/a99b14bd-688d-4f02-b9d5-8ff879c8ed3c" />
<br/>

Only **29 out of 25,000** pairs were mismatched (length ratio outside 0.5–2.0) confirming high-quality alignment between English and Arabic sentences.
  
***The analysis confirmed that dataset is clean, balanced and diverse with strong alignment and a rich vocabulary in both languages. This provides an excellent foundation for training and evaluating English–Arabic machine translation models.***

## Data Preprocessing
<img width="1598" height="1316" alt="image" src="https://github.com/user-attachments/assets/ef659280-0d6e-4d20-ad7d-3f68bcef6657" />
<br/>

- **Initial dataset size:** 25,000 sentence pairs.
- **Text normalization:**  
  - English: Lowercased and standardized whitespace.
  - Arabic: Removed diacritics, unified letter forms (e.g., different forms of 'ا'), normalized punctuation and standardized whitespace.
- **Noise removal:**  
  - Dropped any pairs with missing values or duplicates.
  - Filtered out misaligned pairs based on sentence length ratio (kept only pairs where Arabic/English length ratio is between 0.5 and 2.0).
- **Tokenizer preview:**  
  - Example tokenization output shown for the first English sentence, confirming correct integration with the MarianMT tokenizer.
- **Final dataset size after cleaning:** 24,966 sentence pairs.
- **Data split:**  
  - Training set: 17,476 pairs  
  - Validation set: 3,745 pairs  
  - Test set: 3,745 pairs  
- **Export:**  
  - All splits saved in JSON Lines format, ready for Hugging Face model training.

## Modeling 
<img width="1592" height="997" alt="image" src="https://github.com/user-attachments/assets/42e7ff89-9f85-47be-bb9c-02822970396f" />
<img width="1600" height="1310" alt="image" src="https://github.com/user-attachments/assets/344bc93d-6151-431f-ad9e-ce427b3f0b3d" />
<br/>

- **Model and Tokenizer Initialization:**  
  Loaded the [Helsinki-NLP/opus-mt-en-ar](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) MarianMT model and its tokenizer for English-to-Arabic translation.

- **Data Loading:**  
  Imported the cleaned and preprocessed train, validation and test splits (from preprocessing output)in Hugging Face JSON format using the `datasets` library.

- **Preprocessing Function:**  
  Defined a function to tokenize English source sentences and Arabic target sentences, truncating to a maximum length of 128 tokens. The function aligns input IDs and labels for supervised training.

- **Tokenization:**  
  Applied the preprocessing function to all splits using the `.map()` method, removing original columns and preparing the data for model input.

- **Data Collator:**  
  Used `DataCollatorForSeq2Seq` to dynamically pad batches during training and evaluation, ensuring efficient GPU utilization.

- **Evaluation Metric:**  
  Configured BLEU as the main evaluation metric for machine translation using the `evaluate` library. Implemented a `compute_metrics` function to decode predictions and references then compute BLEU scores for validation and test sets.

- **Training Arguments:**  
  Set up `Seq2SeqTrainingArguments` with:
  - Output directory for checkpoints
  - Evaluation and saving at each epoch
  - Learning rate: 2e-5
  - Batch size: 16
  - Weight decay: 0.01
  - 3 training epochs
  - Mixed precision (fp16) if GPU is available
  - Automatic loading of the best model based on BLEU

- **Trainer Setup:**  
  Initialized a `Seq2SeqTrainer` with the model, arguments, datasets, tokenizer, data collator and metric function.  
  Training is started with `trainer.train()` and the best checkpoint is saved for later evaluation.

This pipeline ensures robust, reproducible fine-tuning of the translation model with automated evaluation and checkpoints.

## Evaluation
<img width="1723" height="553" alt="image" src="https://github.com/user-attachments/assets/74a018ce-5b7a-4490-b14e-8357efb3597c" />

- **BLEU Score (measures how similar each model translation is to the reference translation; higher is better, max 100):**  
 BLEU score is **70.25** on the test set indicating strong overlap between model translations and ground truth (human or reference) translations.
- **ROUGE Score (measures how many words and short phrases the model translation shares with the reference; higher means more similar but ROUGE less useful for very short sentences):**  
  - ROUGE-1: 0.0040
  - ROUGE-2: 0.0003
  - ROUGE-L: 0.0040  
  (Note: ROUGE is less informative for short sentence-level translation tasks. This explains the low score here)
- **METEOR Score (measures how well the model translation matches the reference in meaning and word choice, including synonyms and word order; higher means more similar, max 1.0):**  
  METEOR: **0.8301**  
  This high score reflects strong alignment in meaning and word choice between model outputs and references.

### Visualizations

- **Sentence-level BLEU Scores:**
<img width="850" height="470" alt="image" src="https://github.com/user-attachments/assets/d50d5183-d3c3-4d99-abbf-080164d0e2b6" />
<br/>

  The plot above shows the BLEU score for each sentence in the test set. BLEU measures how similar each model translation is to the reference translation. This helps visualize the consistency and range of translation quality across all examples.
- insights:
  - Many sentences have BLEU scores near 0 or 100, with a wide spread in between.
  - High BLEU scores (close to 100) mean the model's translation is very similar to the reference.
  - Low scores (close to 0) mean the translation is quite different from the reference.
  - The dense vertical lines indicate many sentences share similar scores, possibly due to short sentences or repeated phrases.

- **Qualitative Examples:**
 <img width="989" height="740" alt="image" src="https://github.com/user-attachments/assets/82b2454d-7d5f-40c9-a56f-e430c75e66f6" />
<br/>

  The figure displays several English sentences with both the model's Arabic translation and the ground truth (reference) translation. This allows for direct comparison and qualitative assessment of translation fluency and accuracy.
  Arabic outputs are reshaped and displayed with correct right-to-left order making it easy to compare model and reference translations visually.

- **Attention Map:**
<img width="595" height="469" alt="image" src="https://github.com/user-attachments/assets/1aa99b17-ed46-4a3a-b3a2-be2c0d67607b" />
<br/>
  
  The attention heatmap visualizes how the model aligns English source tokens with Arabic target tokens during translation. Brighter spots indicate stronger attention, helping to interpret which input words the model focuses on for
  
- Insights:
  - Some target tokens strongly attend to specific source tokens (indicated by bright yellow regions).  
  - Other tokens have more diffuse attention, suggesting that the model considers multiple source tokens for generating the translation.  
  - Patterns reveal that the model has learned alignment between English and Arabic sentence structures.  
  - Observing attention maps can help identify translation errors or verify proper word alignment in generated outputs.

## Project structure 
```
Task8/
│
├── data/
│ └── arabic_english_sentences.csv # dataset
│ └── test.json
│ └── train.json
│ └── val.json
│
├── multilingual_translation.ipynb # full notebook
│
├── outputs/
│ └── translated_texts.json # model predictions
│
├── figures/
│ ├── bleu_scores.png
│ ├── attention_maps.png
│ └── translation_examples.png
│
│
└── README.md
```
## How to run 
1. Install requirements:  
   `pip install -r requirements.txt`
2. Download the dataset and place it in the `data/` folder.
3. Open `multilingual_translation.ipynb` in VS Code or Jupyter.
4. Run all cells to preprocess data, train the model, and evaluate results.


## Reference
- [Arabic–English Sentence Bank 25k dataset](https://www.kaggle.com/datasets/tahaalselwii/the-arabic-english-sentence-bank-25k/data)
- [Helsinki-NLP/opus-mt-en-ar](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar)





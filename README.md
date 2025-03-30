# iterative-set-expansion

## Author

Ziyue Jin, UNI: zj2393 (zj2393@columbia.edu)  
Ken Deng, UNI: kd3005 (kd3005@columbia.edu)

---

## List of Files
```
ise.py  
requirements.txt  
README.me
transcript1.txt
transcript2.txt
```

---

## Quick Start

Follow the insturctions on the course website. Place the ise.py in the SpanBERT repository folder. Run the ise.py with proper argument.

### Example Usage
```
python ise.py <Search API> <Search Engine ID> <Gemini API> -- -spanbert 2 0.7 "bill gates microsoft" 10

python ise.py <Search API> <Search Engine ID> <Gemini API> -- -gemini 2 0.0 "bill gates microsoft" 10
```

---


## Quick Start

In a VM that strictly follows the setup instructions (including Python ≥ 3.10 and spaCy + SpanBERT setup) and with the venv activated:

Install required dependencies:
```
pip install -r requirements.txt
```

Run the main script:
```
python ise.py <google_api_key> <google_engine_id> <gemini_api_key> <method> <relation_id> <threshold> "<seed_query>" <k>
```

### Example

```bash
# SpanBERT-based extraction for Work_For relation with confidence ≥ 0.7
python ise.py ABC123XYZ abc456def GEMINIKEY -spanbert 2 0.7 "bill gates microsoft" 10

# Gemini-based extraction for the same relation (threshold ignored)
python ise.py ABC123XYZ abc456def GEMINIKEY -gemini 2 0.0 "bill gates microsoft" 10
```

---

## Engine ID and API Key

will update when submitting

---

## Supported Relations

| ID | Relation Name         | Internal Code                 | Subject Type     | Object Type(s)                             |
|----|-----------------------|-------------------------------|------------------|---------------------------------------------|
| 1  | Schools_Attended      | `per:schools_attended`        | PERSON           | ORGANIZATION                                |
| 2  | Work_For              | `per:employee_of`             | PERSON           | ORGANIZATION                                |
| 3  | Live_In               | `per:cities_of_residence`     | PERSON           | LOCATION, CITY, STATE_OR_PROVINCE, COUNTRY |
| 4  | Top_Member_Employees  | `org:top_members/employees`   | ORGANIZATION     | PERSON                                      |

---

## Detailed Algorithm Description

This project implements **Iterative Set Expansion (ISE)** for relation extraction:

1. Begin with a **seed query** (e.g., `"bill gates microsoft"`).
2. Use the **Google Custom Search API** to retrieve the top-10 webpages.
3. **Extract clean text** from webpages via `BeautifulSoup`.
4. Use **spaCy** to split text into sentences and extract **named entities**.
5. Extract candidate entity pairs and apply:
   - **SpanBERT** classifier (if `-spanbert`)
   - **Google Gemini 2.0 Flash API** (if `-gemini`)
6. Keep tuples matching the target relation with high confidence.
7. If fewer than `k` tuples are extracted, pick an unused tuple from the set as a new query, and repeat.

The process ends when `k` tuples are successfully extracted or all tuples have been used as queries.

---

## Internal Code Design

### High-Level Components

#### 1. Initialization & Argument Parsing

- Parses CLI arguments: API keys, extraction method, target relation ID, threshold, query, and `k`.
- Loads:
  - spaCy's `en_core_web_lg` model
  - SpanBERT model (if `-spanbert`)
  - Configures Gemini API (if `-gemini`)

*Key components:*  
`argparse`, `spacy.load()`, `SpanBERT(...)`, `genai.configure()`

---

#### 2. Web Search & Content Retrieval

- Uses Google API to retrieve top-10 URLs for a query.
- Filters out non-HTML documents (`.pdf`, `.ppt`, etc.)
- Fetches HTML content with `requests`
- Cleans the content to extract pure readable text using `BeautifulSoup`

*Key functions:*  
`search_query()`, `is_likely_html()`, `fetch_full_text()`

---

#### 3. Text Annotation & Candidate Pair Construction

- Uses spaCy to:
  - Split clean text into sentences
  - Identify named entities with types (e.g., PERSON, ORG)
- Pairs are only created between allowed types (depending on target relation)
- Candidate entity pairs are constructed with positional indices for SpanBERT

*Key functions:*  
`get_entities()`, `create_entity_pairs()`

---

#### 4. Relation Extraction Modules

##### A. SpanBERT-Based Extraction
- Filters candidate pairs by type
- Predicts relation label and confidence using `spanbert.predict(...)`
- Accepts only matching relations with confidence ≥ threshold

*Key function:*  
`extract_relations_spanbert(...)`

##### B. Gemini-Based Extraction
- Filters sentences with relevant entity pairs
- Sends each sentence as a prompt to Gemini
- Parses results using format:
```
Subject: X, Object: Y
```
- Adds all extracted tuples with confidence = 1.0

*Key functions:*  
`extract_relations_from_sentence_gemini()`, `extract_relations_gemini()`

---

#### 5. Iterative Set Expansion Logic

- Main loop that:
  - Issues a query
  - Extracts new tuples
  - Adds them to set `X` after removing duplicates
  - If `|X| < k`, uses next unused tuple as a query

*Key logic resides in:*  
`main()`

---

#### 6. Deduplication

- Removes exact duplicate tuples
- For `-spanbert`, keeps the tuple with highest confidence

*Function:*  
`remove_duplicates()`

---

### Gemini-Specific Considerations

- Gemini prompts are carefully constructed to avoid noisy output
- Uses `time.sleep(5)` to avoid rate limits (error 429)
- Always assumes confidence = 1.0 (Gemini does not return scores)

---

## Summary of the Code Structure

1. **ise.py**  
   Main driver script combining all steps:
   - Input parsing
   - Search and extraction
   - Iterative loop for query expansion

2. **spanbert.py**  
   Provides an interface to load and run the SpanBERT model for relation classification.

3. **spacy_help_functions.py**  
   Contains helper functions for:
   - Entity extraction from spaCy sentences
   - Creation of candidate entity pairs

4. **example_relations.py**  
   Sample demonstration script for using SpanBERT.

5. **requirements.txt**  
   Lists all Python packages needed:
   - `spacy`, `beautifulsoup4`, `google-api-python-client`, `requests`, `google-generativeai`, etc.

---

## External Libraries and Roles

| Library | Purpose |
|---------|---------|
| `google-api-python-client` | Access Google Custom Search Engine |
| `requests` | Fetch raw HTML from web pages |
| `BeautifulSoup (bs4)` | Clean HTML and extract visible text |
| `spaCy` | NLP: sentence splitting + named entity recognition |
| `google-generativeai` | Interface to Gemini 2.0 Flash LLM |
| `SpanBERT` | Fine-tuned model for traditional relation classification |

---

## Notes and Caveats

- If a sentence lacks required entity types, it is skipped entirely.
- SpanBERT is computationally expensive — only run it on valid entity pairs.
- Gemini calls may occasionally fail (500/429); retry with delay.
- Confidence threshold is ignored in Gemini mode.
- Only exact duplicates are removed — approximate duplicates (e.g., different case) are retained.

---

## Output Format

For each extracted tuple, the system prints:

```
Subject: <entity1>, Object: <entity2>, Confidence: <score>
```

Example:
```
Subject: Bill Gates, Object: Microsoft, Confidence: 1.00
Subject: Satya Nadella, Object: Microsoft, Confidence: 0.98
...
```

- Transcript 1: Output for `-spanbert 2 0.7 "bill gates microsoft" 10`
- Transcript 2: Output for `-gemini 2 0.0 "bill gates microsoft" 10`

---

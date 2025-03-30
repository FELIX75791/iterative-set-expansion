#!/usr/bin/env python3
"""
ise.py

Iterative Set Expansion for Information Extraction

Usage:
    python ise.py <google_api_key> <google_engine_id> <gemini_api_key> <method> <relation> <threshold> "<seed_query>" <k>

Example (using SpanBERT for Work_For extraction with threshold 0.8):
    python ise.py ABC123XYZ abc456def GEMINIKEY -spanbert 2 0.8 "bill gates microsoft" 10

Arguments:
    google_api_key    : Your Google Custom Search API key (from Project 1)
    google_engine_id  : Your Google Custom Search Engine ID (from Project 1)
    gemini_api_key    : Your Google Gemini API key (see project instructions)
    method            : Extraction method: either -spanbert or -gemini
    relation          : Integer (1-4) indicating the target relation:
                           1 -> per:schools_attended
                           2 -> per:employee_of
                           3 -> per:cities_of_residence
                           4 -> org:top_members/employees
    threshold         : Extraction confidence threshold (between 0 and 1; ignored for -gemini)
    seed_query        : Seed query (a plausible tuple, in quotes, e.g., "bill gates microsoft")
    k                 : Desired number of extracted tuples

The script performs iterative search-and-extract: it starts with the seed query,
retrieves top-10 webpages via the Google Custom Search API, extracts plain text via BeautifulSoup,
annotates text with spaCy, and then extracts candidate tuples using either the pre-trained SpanBERT
classifier or the Google Gemini API. It adds high-confidence tuples to the extraction set X, then
uses an unused tuple to generate a new query if k tuples have not been collected.
"""

import sys
import argparse
import requests
from bs4 import BeautifulSoup
import spacy
import time
from googleapiclient.discovery import build
import google.generativeai as genai
import re

# Try to import SpanBERT if using -spanbert
try:
    from spanbert import SpanBERT
except ImportError:
    SpanBERT = None

# Mapping between spaCy entity labels and internal types used by SpanBERT.
spacy2bert = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE"
}

bert2spacy = {
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "CITY": "GPE",
    "COUNTRY": "GPE",
    "STATE_OR_PROVINCE": "GPE",
    "DATE": "DATE"
}

# Relation specifications: for each relation number, the internal relation name and required entity types.
relation_specs = {
    1: {"name": "per:schools_attended", "subj": "PERSON", "obj": "ORGANIZATION"},
    2: {"name": "per:employee_of", "subj": "PERSON", "obj": "ORGANIZATION"},
    3: {"name": "per:cities_of_residence", "subj": "PERSON", "obj": ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]},
    4: {"name": "org:top_members/employees", "subj": "ORGANIZATION", "obj": "PERSON"}
}

# File extensions we consider non-HTML.
NON_HTML_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}


def is_likely_html(url):
    """Return True if the URL is likely to point to an HTML page."""
    url = url.lower()
    for ext in NON_HTML_EXTENSIONS:
        if url.endswith(ext):
            return False
    return True

def clean_text(text):
    """Remove zero-width spaces and other non-visible characters."""
    return re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)

def fetch_full_text(url):
    """Fetch full text from URL by retrieving HTML and extracting text via BeautifulSoup."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code != 200:
            return ""
        html = response.text
        # Try using lxml if available, otherwise fallback to html.parser
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        # Remove script and style tags
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        # Extract text from paragraphs, divs, etc.
        texts = soup.stripped_strings
        full_text = " ".join(texts)
        full_text = clean_text(full_text)
        return full_text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""



def search_query(service, engine_id, query, num_results=10):
    """
    Execute the query via the Google Custom Search API.
    Returns a list of tuples: (title, link, snippet, is_html).
    """
    try:
        res = service.cse().list(q=query, cx=engine_id, num=num_results).execute()
        items = res.get("items", [])
        results = []
        for item in items:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            html_flag = is_likely_html(link)
            results.append((title, link, snippet, html_flag))
        return results
    except Exception as e:
        print(f"Error in search query: {e}")
        return []


# Helper functions from spacy_help_functions.py
def get_entities(sentence):
    """Extract entities from a spaCy sentence and map their labels."""
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    """
    Create candidate entity pairs from a spaCy sentence object.
    entities_of_interest is a list of internal types (e.g., "ORGANIZATION", "PERSON", etc.)
    """
    entities_of_interest_set = {bert2spacy[e] for e in entities_of_interest}
    ents = sents_doc.ents
    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if e1.label_ not in entities_of_interest_set:
            continue
        for j in range(i + 1, len(ents)):
            e2 = ents[j]
            if e2.label_ not in entities_of_interest_set:
                continue
            if e1.text.lower() == e2.text.lower():
                continue
            if (1 <= (e2.start - e1.end) <= window_size):
                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token and start >= 0:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                    left_r = start + 2 if start >= 0 else 0
                else:
                    left_r = 0
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token and start < length_doc:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc
                if (right_r - left_r) > window_size:
                    continue
                tokens = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                # Add both orders
                entity_pairs.append({"tokens": tokens, "subj": e1_info, "obj": e2_info})
                entity_pairs.append({"tokens": tokens, "subj": e2_info, "obj": e1_info})
    return entity_pairs

def is_valid_string(s):
    """Return True if s is a valid non-empty string (after stripping) and does not start with a zero-width space."""
    return bool(s.strip()) and not s.startswith('\u200b')

# Extraction using SpanBERT
def extract_relations_spanbert(text, nlp, spanbert, relation_spec, threshold):
    """
    Given plain text, use spaCy and SpanBERT to extract candidate tuples for the target relation.
    Only candidate pairs with the correct entity types are considered.
    """
    results = []
    doc = nlp(text)
    for sentence in doc.sents:
        candidate_pairs = create_entity_pairs(
            sentence, ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "DATE"]
        )
        filtered_pairs = []
        for pair in candidate_pairs:
            subj_type = pair["subj"][1]
            obj_type = pair["obj"][1]
            required_subj = relation_spec["subj"]
            required_obj = relation_spec["obj"]
            if isinstance(required_obj, list):
                condition = (subj_type == required_subj) and (obj_type in required_obj)
            else:
                condition = (subj_type == required_subj) and (obj_type == required_obj)
            if condition:
                filtered_pairs.append(pair)
        if not filtered_pairs:
            continue
        predictions = spanbert.predict(filtered_pairs)
        for pair, (rel, conf) in zip(filtered_pairs, predictions):
            subj_text = pair["subj"][0]
            obj_text = pair["obj"][0]
            if rel == relation_spec["name"] and conf >= threshold:
                if is_valid_string(subj_text) and is_valid_string(obj_text):
                    results.append((subj_text, obj_text, conf))
    return results


# Extraction using Gemini
def get_gemini_completion(prompt, model_name="gemini-2.0-flash", max_tokens=200, temperature=0.2, top_p=1, top_k=32):
    """
    Invoke the Gemini API with the given prompt.
    """
    time.sleep(5)
    model = genai.GenerativeModel(model_name)
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text.strip() if response.text else "No response received"


def extract_relations_from_sentence_gemini(sentence, relation_spec):
    """
    For a single sentence, call Gemini to extract tuples for the target relation.
    The prompt instructs Gemini to output tuples in the format:
      Subject: <subject>, Object: <object>
    """
    if isinstance(relation_spec["obj"], list):
        required_obj = " or ".join(relation_spec["obj"])
    else:
        required_obj = relation_spec["obj"]
    prompt = (
        f"Extract all instances of the relation '{relation_spec['name']}' from the following sentence. "
        f"Only extract if the subject is of type {relation_spec['subj']} and the object is of type {required_obj}. "
        "Return each tuple in the format 'Subject: <subject>, Object: <object>' on a separate line. "
        "If no relation is found, just return 'None'.\n"
        f"Sentence: {sentence}"
    )
    response = get_gemini_completion(prompt)
    results = []
    if response.strip().lower() == "none" or response.strip() == "":
        return results
    lines = response.splitlines()
    for line in lines:
        if "Subject:" in line and "Object:" in line:
            try:
                parts = line.split("Subject:")[1].split("Object:")
                subj = parts[0].strip().rstrip(",")
                obj = parts[1].strip()
                # For Gemini, we assume valid output so no additional filtering here.
                results.append((subj, obj, 1.0))
            except Exception:
                continue
    return results


def extract_relations_gemini(text, nlp, relation_spec):
    """
    Given plain text, use spaCy to select candidate sentences and then use Gemini to extract tuples.
    """
    results = []
    doc = nlp(text)
    for sentence in doc.sents:
        candidate_pairs = create_entity_pairs(
            sentence, ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
        )
        valid = False
        for pair in candidate_pairs:
            subj_type = pair["subj"][1]
            obj_type = pair["obj"][1]
            required_subj = relation_spec["subj"]
            required_obj = relation_spec["obj"]
            if isinstance(required_obj, list):
                condition = (subj_type == required_subj) and (obj_type in required_obj)
            else:
                condition = (subj_type == required_subj) and (obj_type == required_obj)
            if condition:
                valid = True
                break
        if valid:
            rels = extract_relations_from_sentence_gemini(str(sentence), relation_spec)
            results.extend(rels)
    return results



# To remove duplicate tuples
def remove_duplicates(tuples_list, use_confidence=True):
    unique = {}
    for tup in tuples_list:
        key = (tup[0].lower(), tup[1].lower())
        if key not in unique or (use_confidence and tup[2] > unique[key][2]):
            unique[key] = tup
    return list(unique.values())

def is_valid_tuple(tup):
    subj, obj, _ = tup
    return is_valid_string(subj) and is_valid_string(obj)

# Main iterative procedure
def main():
    parser = argparse.ArgumentParser(description="Iterative Set Expansion for Information Extraction")
    parser.add_argument("google_api_key", help="Google Custom Search API Key")
    parser.add_argument("google_engine_id", help="Google Custom Search Engine ID")
    parser.add_argument("gemini_api_key", help="Google Gemini API Key")
    parser.add_argument("method", choices=["-spanbert", "-gemini"], help="Extraction method: -spanbert or -gemini")
    parser.add_argument("relation", type=int, choices=[1, 2, 3, 4], help="Relation to extract (1-4)")
    parser.add_argument("threshold", type=float, help="Extraction confidence threshold (ignored for -gemini)")
    parser.add_argument("seed_query", help="Seed query (in quotes, e.g., \"bill gates microsoft\")")
    parser.add_argument("k", type=int, help="Number of tuples to extract")
    args = parser.parse_args()

    # Configure Gemini API key.
    genai.configure(api_key=args.gemini_api_key)

    # Build the Custom Search service.
    service = build("customsearch", "v1", developerKey=args.google_api_key)

    # Load spaCy language model.
    nlp = spacy.load("en_core_web_lg")

    # If using SpanBERT, load the pre-trained model.
    if args.method == "-spanbert":
        if SpanBERT is None:
            print("Error: SpanBERT module not found. Make sure you have the SpanBERT code in your working directory.")
            sys.exit(1)
        spanbert = SpanBERT("./pretrained_spanbert")
    else:
        spanbert = None

    relation_spec = relation_specs[args.relation]
    threshold = args.threshold
    seed_query = args.seed_query
    k = args.k

    # Initialize extraction set and bookkeeping.
    X = []  # list of tuples: (subject, object, confidence)
    used_queries = set()
    query_queue = [seed_query]
    processed_urls = set()

    # Iterative search-and-extract loop.
    while query_queue and len(X) < k:
        current_query = query_queue.pop(0)
        if current_query in used_queries:
            continue
        used_queries.add(current_query)
        print(f"\nQuerying: {current_query}")
        results = search_query(service, args.google_engine_id, current_query, num_results=10)
        if not results:
            print("No results returned for query.")
            continue
        for title, link, snippet, is_html in results:
            if link in processed_urls:
                print(f"Already processed url: {link}")
                continue
            processed_urls.add(link)
            if not is_html:
                print(f"Not html: {link}")
                continue
            text = fetch_full_text(link)
            if not text:
                print(f"No text from the page: {link}")
                continue
            if len(text) > 10000:
                text = text[:10000]
            # Extract tuples from the webpage.
            if args.method == "-spanbert":
                extracted = extract_relations_spanbert(text, nlp, spanbert, relation_spec, threshold)
            else:
                extracted = extract_relations_gemini(text, nlp, relation_spec)
            if extracted:
                print(f"Extracted from {link}: {extracted}")
            else:
                print(f"Nothing extracted from {link}")
            X.extend(extracted)
            # Remove exact duplicates (keeping highest confidence if applicable).
            X = remove_duplicates(X, use_confidence=(args.method == "-spanbert"))
            if len(X) >= k:
                break
        if len(X) >= k:
            break
        # If not enough tuples, select an unused tuple from X to form a new query.
        # For -spanbert, choose the tuple with the highest confidence.
        remaining = [t for t in X if " ".join(t[:2]) not in used_queries]
        if not remaining and X:
            new_query = " ".join(X[0][:2])
            if new_query not in used_queries:
                query_queue.append(new_query)
        elif remaining:
            if args.method == "-spanbert":
                remaining.sort(key=lambda x: x[2], reverse=True)
            new_query = " ".join(remaining[0][:2])
            if new_query not in used_queries:
                query_queue.append(new_query)
    # Final output: if using SpanBERT, sort by confidence.
    if args.method == "-spanbert":
        X.sort(key=lambda x: x[2], reverse=True)
    if len(X) > k:
        X = X[:k]
    # Filter out any invalid tuples before printing.
    X = list(filter(is_valid_tuple, X))
    print("\n==================== Extracted Tuples ====================")
    for tup in X:
        print(f"Subject: {tup[0]}, Object: {tup[1]}, Confidence: {tup[2]:.2f}")


if __name__ == "__main__":
    main()

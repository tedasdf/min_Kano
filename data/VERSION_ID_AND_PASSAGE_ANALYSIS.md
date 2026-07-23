Yes — your dataset analysis should focus on **two separate questions**:

1. Does `version_id` correctly group passages belonging to one document?
2. Can the same underlying document appear under different `version_id` values?

## What does this ID mean?

```text
nsw_caselaw:549fc6183004262463bb648a
```

Break it into:

```text
nsw_caselaw
:
549fc6183004262463bb648a
```

| Part                       | Likely meaning                                                          |
| -------------------------- | ----------------------------------------------------------------------- |
| `nsw_caselaw`              | The source/provider                                                     |
| `549fc6183004262463bb648a` | The provider-specific identifier for that version of the legal document |

The Open Australian Legal Corpus describes `version_id` as a unique identifier for the latest known **version of a document**, not as a passage identifier. The suffix also appears directly in the NSW Caselaw decision URL:

```text
https://www.caselaw.nsw.gov.au/decision/549fc6183004262463bb648a
```

Therefore, for your example:

```text
version_id = document-version ID
source.text = passage/snippet from that document
```

It is **not**:

```text
version_id = passage ID
```

The corpus creator introduced `version_id` partly to track documents that change over time, which also explains the word “version.” ([Hugging Face][1])

## Do not infer its meaning from duplicate counts

This reasoning would be incorrect:

```text
High number of repeated IDs → document ID
Low number of repeated IDs → passage ID
```

The duplicate frequency only tells you how many QA examples were generated from that document in your particular dataset.

For example:

```text
version_id = document_A
```

could occur:

* once, because only one passage/question was sampled;
* 5 times, because five questions were generated;
* 50 times, because many passages were sampled.

It remains a document-version ID in every case.

So calculate duplication frequency, but use it to understand:

```text
QA examples per document
```

not to determine the field’s meaning.

---

# First analysis: does `version_id` group passages correctly? [DONE] 

Group the rows by `source.version_id`.

For each ID, inspect:

```text
version_id
number of rows
number of unique source.text passages
number of unique questions
unique citations
unique URLs
```

You would expect one `version_id` to normally have:

```text
one citation
one URL
possibly many passages
possibly many questions
```

Example:

| `version_id` | Rows | Unique passages | Unique questions | Citations | URLs |
| ------------ | ---: | --------------: | ---------------: | --------: | ---: |
| Document A   |   10 |               4 |               10 |         1 |    1 |
| Document B   |    1 |               1 |                1 |         1 |    1 |
| Document C   |    8 |               3 |                8 |         2 |    2 |

Document C would need investigation because one document ID is connected to multiple citations or URLs.

This check answers:

> Does a single `version_id` consistently represent one parent legal document?

---

# Second analysis: duplicate documents under different IDs

This is the opposite direction.

Instead of asking:

```text
Does one ID point to several documents?
```

you ask:

```text
Do several IDs point to the same document?
```

## Check 1: Repeated URL under different IDs [Done]

Group by normalized URL:

```text
https://www.caselaw.nsw.gov.au/decision/ABC
https://www.caselaw.nsw.gov.au/decision/ABC/
```

After removing trailing slashes, query parameters and other superficial differences:

```text
same URL + different version_id
```

is a strong duplicate candidate.

## Check 2: Repeated own citation under different IDs [done]

Group by normalized `source.citation`.

For example:

```text
Nasr v NRMA Insurance [2006] NSWSC 1018
```

Normalize case and whitespace:

```text
nasr v nrma insurance [2006] nswsc 1018
```

Then look for:

| Normalized citation | Number of IDs | Number of providers |
| ------------------- | ------------: | ------------------: |
| `[2006] NSWSC 1018` |             1 |                   1 |
| `[2011] NSWCA 50`   |             2 |                   2 |

The second row may represent:

* the same judgment mirrored by two providers;
* two versions of the judgment;
* a duplicate ingestion record.

Because the dataset’s `citation` field is intended to describe the document itself, repeated own citations are much stronger evidence than merely finding a citation inside the document text. ([Hugging Face][1])

For judgments, extracting the **neutral citation** is often more reliable than matching the entire case title:

```text
[2006] NSWSC 1018
```

Party names may vary in capitalization, spacing or anonymisation while the neutral citation remains the same.


URL analysis found no duplicate source pages under different version_id values. However, duplicate or mirrored documents may still exist under different URLs and providers.


## Check 3: Same metadata under different IDs [done]

Compare combinations such as:

```text
normalized citation
+ jurisdiction
+ document type
+ publication year/date
```

Possible duplicate candidate:

| ID   | Citation                                | Jurisdiction | Type     |
| ---- | --------------------------------------- | ------------ | -------- |
| ID A | Nasr v NRMA Insurance [2006] NSWSC 1018 | NSW          | decision |
| ID B | NASR v NRMA INSURANCE [2006] NSWSC 1018 | NSW          | decision |

These are almost certainly the same judgment despite formatting differences.

## Check 4: Text similarity

This is the strongest content check, but your current QA dataset may only contain **passages**, not complete documents.

That distinction matters.

### Best case: full document text is available

For each `version_id`:

```text
normalize the entire document text
→ calculate exact hash
→ calculate near-duplicate similarity
```

Then detect:

```text
different version_id
+
same or nearly same full document text
```

The Open Australian Legal Corpus is document-based, while the QA dataset uses snippets taken from those documents. Ideally, duplicate-document analysis should therefore use the original corpus document text, not just the QA snippets. ([Hugging Face][2])

### If only QA passages are available

Combine all unique passages belonging to each ID:

```text
version_id A
    passage 1
    passage 2
    passage 3
```

Then compare those passage collections.

However, this can only produce **duplicate candidates**, not definitive proof. Two copies of the same document may have completely different snippets sampled into the QA dataset, so their passage collections might not overlap at all.

---

# Passage overlap analysis

This is separate from duplicate-document analysis.

You want to identify:

```text
exact same passage
near-duplicate passage
partially overlapping passage
different passage from same document
```

## 1. Exact passage duplicates

Normalize each `source.text` by:

* lowercasing where appropriate;
* standardizing whitespace;
* removing repeated line breaks;
* possibly standardizing quotation characters.

Then group by normalized text.

Report:

| Passage group | Rows | Questions | Document IDs |
| ------------- | ---: | --------: | -----------: |
| Passage A     |    5 |         5 |            1 |
| Passage B     |    2 |         2 |            2 |

Interpretation:

* same passage, same document ID: probably multiple questions generated from one passage;
* same passage, different document IDs: possible duplicate documents or duplicated ingestion.

## 2. Passage containment

Example:

```text
Passage A:
The plaintiff filed the summons seven months after the Local Court decision.

Passage B:
The court noted that the plaintiff filed the summons seven months after
the Local Court decision. No explanation was provided for the delay.
```

Passage A is almost entirely contained inside Passage B.

These are not exact duplicates, but they clearly overlap.

Measure something like:

```text
shared tokens / tokens in shorter passage
```

If almost all of the shorter passage occurs in the longer passage, flag it as containment overlap.

## 3. Partial overlap

Example:

```text
Passage A: paragraphs 10–20
Passage B: paragraphs 17–27
```

Neither passage fully contains the other.

For this, use token shingles, such as sequences of five consecutive tokens:

```text
passage → set of 5-token shingles
```

Then compare the shingle sets using Jaccard similarity:

[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
]

High similarity suggests near-duplication or overlapping extraction.

You do not need to select a perfect threshold immediately. First inspect similarity bands, for example:

| Similarity | Interpretation to inspect                 |
| ---------: | ----------------------------------------- |
|       1.00 | Exact after normalization                 |
|  0.80–0.99 | Near duplicate or extended/shortened copy |
|  0.40–0.80 | Likely partial overlap                    |
| Below 0.40 | Possibly distinct passages                |

These are exploratory ranges, not universal legal-dataset rules.

---

# Where to perform the overlap checks

Perform them at two levels.

## Within the same `version_id`

This tells you about the dataset’s passage construction:

```text
same document
→ exact repeated passages?
→ overlapping passages?
→ distinct passages?
```

This is not automatically split leakage if the entire document remains in one split.

It helps you understand:

* questions per passage;
* whether sliding windows were used;
* whether snippets were duplicated;
* whether false negatives may arise.

## Across different `version_id` values

This helps detect:

* duplicate documents under different IDs;
* mirrored judgments;
* corrected versions;
* extraction duplicates;
* accidentally reused passages.

This is more directly relevant to hidden leakage.

---

# The analysis structure I recommend

## A. Validate the meaning and consistency of `version_id`

| Check                   | What it reveals                                       |
| ----------------------- | ----------------------------------------------------- |
| Rows per `version_id`   | QA examples per document                              |
| Unique passages per ID  | Passages sampled per document                         |
| Unique citations per ID | Whether one ID maps consistently to one document      |
| Unique URLs per ID      | Whether one ID maps consistently to one source record |

## B. Search for duplicate documents across IDs

| Check                                    | Strength                        |
| ---------------------------------------- | ------------------------------- |
| Same normalized URL, different IDs       | Very strong                     |
| Same own neutral citation, different IDs | Very strong                     |
| Same citation/title/date/court           | Strong                          |
| Same normalized full-text hash           | Conclusive for exact duplicates |
| High full-document text similarity       | Strong for near duplicates      |
| Similar QA passages only                 | Candidate evidence              |

## C. Analyse passage relationships

| Relationship                    | Meaning                                           |
| ------------------------------- | ------------------------------------------------- |
| Same text, same ID              | Several questions from one passage                |
| Same text, different IDs        | Possible hidden document duplicate                |
| Overlapping text, same ID       | Passage/chunk construction                        |
| Overlapping text, different IDs | Possible mirrored or duplicate document           |
| Different text, same ID         | Different passages from one document              |
| Different text, different IDs   | Potentially clean, pending document deduplication |

## The main correction to your current thinking

Do not determine whether `version_id` is a document ID by counting how often it repeats.

The field definition and URL structure already tell you it represents the **document version**. Your dataset analysis should instead test:

> Is this identifier consistently used, and can one underlying judgment still appear under several different identifiers?

That gives you a sensible sequence:

```text
1. Treat version_id as the supplied document-version key.
2. Check whether one ID maps to consistent metadata.
3. Find repeated URLs and own citations under different IDs.
4. Detect exact and near-duplicate document text where available.
5. Analyse exact and partial passage overlaps.
6. Build document families before producing train/validation/test splits.
```

[1]: https://huggingface.co/datasets/isaacus/open-australian-legal-corpus?utm_source=chatgpt.com "isaacus/open-australian-legal-corpus · Datasets at ..."
[2]: https://huggingface.co/datasets/isaacus/open-australian-legal-qa?utm_source=chatgpt.com "isaacus/open-australian-legal-qa · Datasets at Hugging Face"

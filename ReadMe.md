# Mini Kanon 3

## Project Overview

Mini Kanon 3 is a compact, multi-capability legal language model inspired by Isaacus's Kanon model family.

The project is designed as an **ML engineering portfolio project**, not as an attempt to reproduce Isaacus's proprietary frontier model or make a new research claim.

The goal is to build, evaluate, and deploy one legal-text system that supports four capabilities through a consistent interface:

1. **Embed**
2. **Classify**
3. **Enrich**
4. **Segment**

---

## Primary Deliverable

The final deliverable is an end-to-end legal document intelligence system with:

- a shared pretrained text encoder;
- task-specific components for embedding, classification, enrichment, and segmentation;
- reproducible data-processing and training pipelines;
- task-specific evaluation suites;
- one unified inference API;
- experiment tracking and error analysis;
- automated tests;
- containerised deployment;
- technical documentation and a demonstration application.

The project should demonstrate practical ML engineering skills across the full model lifecycle:

```text
Data preparation
      ↓
Baseline models
      ↓
Training and evaluation
      ↓
Multi-capability integration
      ↓
API and deployment
      ↓
Testing, monitoring, and documentation
```

---

## Capability 1: Embed

### Goal

Convert legal queries, clauses, passages, and documents into dense vector representations for semantic retrieval.

### Input

- legal query;
- legal passage or document segment.

### Output

- fixed-size embedding vector.

### Intended Uses

- semantic search;
- retrieval-augmented generation;
- similar-clause retrieval;
- document clustering;
- duplicate or near-duplicate detection.

### Minimum Deliverable

- prepare positive and negative query-passage pairs;
- fine-tune or adapt a pretrained embedding model;
- implement vector indexing and retrieval;
- compare the trained model against a pretrained baseline;
- provide a retrieval demonstration.

### Evaluation Metrics

- Recall@1;
- Recall@5;
- Recall@10;
- Mean Reciprocal Rank;
- NDCG@10;
- retrieval latency;
- embedding throughput.

### Acceptance Criteria

The embedding system must:

- return a fixed-size vector for every valid input;
- retrieve legally relevant passages above unrelated passages;
- outperform or meaningfully match the selected baseline on at least one primary retrieval metric;
- support batched inference;
- integrate with the unified API.

---

## Capability 2: Classify

### Goal

Determine whether a legal text supports a user-provided natural-language statement.

This is a **statement-based classifier**, not a fixed multiclass classifier.

### Input

```json
{
  "text": "The supplier must keep all customer information confidential.",
  "statement": "This is a confidentiality clause."
}
```

### Output

```json
{
  "score": 0.96,
  "label": true
}
```

### Core Formulation

```text
legal text + natural-language statement
                    ↓
          probability or support score
```

### Initial Statement Templates

The initial evaluation set may include statements covering:

- confidentiality;
- governing law;
- indemnity;
- payment;
- termination;
- termination for convenience;
- limitation of liability;
- liability cap;
- intellectual-property assignment or licence;
- non-compete;
- representation or warranty;
- force majeure;
- legal obligations;
- legal rights;
- clauses applying to a named party.

These templates are used for training and evaluation, but the model interface must allow a user to submit a new statement at inference time.

### Minimum Deliverable

- construct positive and negative text-statement pairs;
- train or fine-tune a binary statement-pair classifier;
- output a calibrated support score;
- select and document a classification threshold;
- evaluate generalisation to statement wording not seen during training.

### Evaluation Metrics

- precision;
- recall;
- F1 score;
- AUROC;
- average precision;
- Brier score or expected calibration error;
- per-template performance;
- false-positive and false-negative analysis.

### Acceptance Criteria

The classifier must:

- accept arbitrary natural-language statements;
- return a score between `0` and `1`;
- support multiple statements for the same text;
- demonstrate performance beyond a simple keyword baseline;
- report calibrated confidence or clearly document calibration limitations;
- integrate with the unified API.

---

## Capability 3: Enrich

### Goal

Transform unstructured legal text into structured, machine-readable information according to a user-provided schema.

### Input

- legal text;
- custom output schema.

Example schema:

```json
{
  "parties": ["string"],
  "effective_date": "date | null",
  "governing_law": "string | null",
  "obligations": [
    {
      "party": "string",
      "obligation": "string"
    }
  ]
}
```

### Output

```json
{
  "parties": ["Example Pty Ltd", "Customer Pty Ltd"],
  "effective_date": "2026-07-01",
  "governing_law": "Victoria, Australia",
  "obligations": [
    {
      "party": "Example Pty Ltd",
      "obligation": "Maintain the confidentiality of customer information."
    }
  ]
}
```

Where possible, each extracted value should include:

- source text;
- character offsets;
- confidence score;
- validation status.

### Initial Scope

The first version should support a constrained schema rather than unrestricted arbitrary extraction.

Recommended initial fields:

- parties;
- dates;
- governing law;
- monetary amounts;
- obligations;
- rights;
- termination conditions;
- referenced legislation or cases.

### Minimum Deliverable

- define a schema representation;
- validate requested fields;
- extract structured values from legal text;
- preserve source spans where possible;
- validate generated output against the schema;
- return explicit `null` or empty values when information is absent.

### Evaluation Metrics

- exact-match accuracy;
- field-level precision, recall, and F1;
- span-level F1;
- JSON validity rate;
- schema compliance rate;
- hallucination rate;
- missing-value accuracy.

### Acceptance Criteria

The enrichment system must:

- accept at least one documented custom schema format;
- produce machine-readable output;
- pass schema validation;
- avoid inventing unavailable values where possible;
- link extracted values back to their source spans;
- integrate with the unified API.

---

## Capability 4: Segment

### Goal

Identify the structural hierarchy of a legal document and return document sections with their boundaries and parent-child relationships.

### Example Structures

- document title;
- preamble;
- recitals;
- headings;
- sections;
- clauses;
- subclauses;
- paragraphs;
- definitions;
- numbered and bulleted list items;
- schedules;
- signature blocks.

### Input

- raw legal document text.

### Output

```json
{
  "segments": [
    {
      "id": "section-1",
      "type": "section",
      "title": "1. Definitions",
      "start": 0,
      "end": 420,
      "parent_id": null
    },
    {
      "id": "clause-1.1",
      "type": "clause",
      "title": "1.1 Confidential Information",
      "start": 24,
      "end": 210,
      "parent_id": "section-1"
    }
  ]
}
```

### Minimum Deliverable

- detect structural boundaries;
- assign segment types;
- preserve character offsets;
- reconstruct parent-child relationships;
- return segments in document order;
- handle common numbering patterns.

### Evaluation Metrics

- boundary precision, recall, and F1;
- segment-type classification F1;
- hierarchy accuracy;
- exact-span accuracy;
- document coverage;
- overlap and gap error rates.

### Acceptance Criteria

The segmentation system must:

- preserve the original document order;
- return valid, non-negative character offsets;
- avoid overlapping segments unless explicitly allowed;
- represent nested sections;
- recover common legal numbering structures;
- integrate with the unified API.

---

## Proposed System Architecture

The initial architecture should prioritise engineering clarity and manageable scope.

```text
                        ┌─────────────────────┐
                        │ Legal text input    │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │ Shared text encoder │
                        └──────┬────┬────┬────┘
                               │    │    │
                 ┌─────────────┘    │    └──────────────┐
                 │                  │                   │
        ┌────────▼───────┐  ┌──────▼────────┐  ┌──────▼────────┐
        │ Embedding head │  │ Classifier     │  │ Token/span    │
        │                │  │ or pair head   │  │ representations│
        └────────────────┘  └───────────────┘  └──────┬────────┘
                                                       │
                                      ┌────────────────┴──────────────┐
                                      │                               │
                              ┌───────▼────────┐              ┌──────▼───────┐
                              │ Enrichment     │              │ Segmentation │
                              │ component      │              │ component    │
                              └────────────────┘              └──────────────┘
```

The final implementation does not need to force every task into a single neural-network checkpoint.

A valid engineering design may use:

- one shared encoder;
- several lightweight task heads;
- deterministic parsing for obvious document structure;
- schema validation and post-processing;
- separate task-specific checkpoints behind one service.

The important requirement is a consistent, integrated system.

---

## Unified API

### Suggested Endpoint

```http
POST /v1/process
```

### Example Request

```json
{
  "task": "classify",
  "text": "Either party may terminate this agreement with 30 days' notice.",
  "statement": "This clause permits termination for convenience."
}
```

### Supported Tasks

```text
embed
classify
enrich
segment
```

### Additional Endpoints

```http
GET  /health
GET  /version
GET  /metrics
POST /v1/embed
POST /v1/classify
POST /v1/enrich
POST /v1/segment
```

### API Requirements

- typed request and response models;
- clear validation errors;
- batch support where practical;
- versioned model metadata;
- latency measurement;
- structured logging;
- deterministic error handling.

---

## Dataset Deliverables

Each capability must include a documented dataset pipeline.

### Required Components

- raw-data ingestion;
- cleaning and normalisation;
- train, validation, and test splitting;
- duplicate detection;
- leakage checks;
- span-offset preservation;
- class or template distribution analysis;
- negative-sample generation;
- dataset versioning;
- reproducible preprocessing scripts.

### Data Cards

Each dataset should document:

- source;
- licence;
- intended use;
- sample count;
- label definitions;
- known biases;
- known quality issues;
- train-validation-test split strategy;
- excluded data.

---

## Evaluation Deliverables

The project must include a separate evaluation module for every capability.

### Required Outputs

- baseline comparison;
- primary metric table;
- per-category analysis;
- confidence intervals or repeated-run variation where practical;
- error examples;
- latency and throughput measurements;
- memory usage;
- known failure modes.

### End-to-End Evaluation

The integrated system should also be tested on complete legal documents.

An end-to-end test may:

1. segment a contract;
2. embed each segment;
3. retrieve segments relevant to a query;
4. classify selected segments against legal statements;
5. enrich relevant segments into a structured schema.

---

## Engineering Deliverables

### Repository

The repository should include:

```text
mini-kanon-3/
├── configs/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/
│   ├── api/
│   ├── classification/
│   ├── embedding/
│   ├── enrichment/
│   ├── segmentation/
│   ├── evaluation/
│   └── common/
├── tests/
├── Dockerfile
├── pyproject.toml
├── README.md
├── PROJECT.md
└── MODEL_CARD.md
```

### Training

- configuration-driven experiments;
- fixed random seeds;
- checkpoint saving;
- early stopping where appropriate;
- experiment tracking;
- reproducible commands;
- device selection for CPU or GPU.

### Testing

- unit tests for preprocessing;
- unit tests for metrics;
- schema-validation tests;
- API tests;
- model-loading tests;
- malformed-input tests;
- integration tests across all four capabilities.

### Deployment

- Docker image;
- local inference instructions;
- health-check endpoint;
- model-version endpoint;
- CPU fallback;
- documented hardware requirements;
- example requests and responses.

---

## Demonstration Deliverable

The final demonstration should show one legal document passing through the complete system.

### Suggested Demonstration Flow

1. Upload or paste a legal agreement.
2. Detect its sections and clauses.
3. Search for clauses relevant to a legal query.
4. test selected clauses against natural-language statements.
5. Extract requested information using a custom schema.
6. Display source spans and confidence scores.
7. Export the result as JSON.

The demonstration may be implemented as:

- a small web application;
- an interactive notebook;
- a command-line interface;
- or a documented API walkthrough.

A lightweight web interface is preferred for portfolio presentation.

---

## Project Phases

### Phase 1 — Specification and Baselines

Deliverables:

- final task definitions;
- dataset selection;
- input and output schemas;
- baseline embedding model;
- keyword classification baseline;
- rule-based segmentation baseline;
- enrichment schema prototype.

### Phase 2 — Embedding System

Deliverables:

- query-passage dataset;
- negative sampling;
- trained embedding model;
- retrieval index;
- evaluation report;
- retrieval demo.

### Phase 3 — Universal Classification

Deliverables:

- text-statement pair dataset;
- trained classifier;
- probability calibration;
- template and paraphrase evaluation;
- classification error report.

### Phase 4 — Segmentation

Deliverables:

- legal structure dataset;
- boundary detector;
- segment-type predictions;
- hierarchy reconstruction;
- segmentation evaluation report.

### Phase 5 — Enrichment

Deliverables:

- initial supported schema;
- structured extraction pipeline;
- source-span linking;
- schema validation;
- enrichment evaluation report.

### Phase 6 — Integration and Deployment

Deliverables:

- unified inference service;
- automated tests;
- Docker deployment;
- model and data cards;
- end-to-end demonstration;
- final technical report.

---

## Suggested Timeline

| Week | Focus | Main Output |
|---|---|---|
| 1 | Scope, datasets, and baselines | Final specification and baseline results |
| 2 | Embedding data and training | Searchable legal embedding system |
| 3 | Classification data and training | Statement-based legal classifier |
| 4 | Segmentation | Hierarchical legal document parser |
| 5 | Enrichment | Schema-driven structured extraction |
| 6 | Integration | Unified API and end-to-end tests |
| 7 | Deployment and optimisation | Docker image, latency benchmarks, demo |
| 8 | Documentation and portfolio polish | Final report, model card, diagrams, README |

---

## Final Portfolio Outputs

The completed project should contain:

1. public code repository;
2. trained model checkpoints or reproducible training instructions;
3. dataset-processing scripts;
4. evaluation reports for all capabilities;
5. unified inference API;
6. Docker deployment;
7. demonstration application;
8. architecture diagram;
9. model card;
10. data cards;
11. technical blog post;
12. short demonstration video.

---

## Non-Goals

The initial version will not attempt to:

- reproduce Isaacus's proprietary datasets or architecture;
- claim frontier legal-model performance;
- provide legal advice;
- replace professional legal review;
- support every jurisdiction or document type;
- guarantee unrestricted arbitrary-schema extraction;
- train a large foundation model from scratch.

---

## Definition of Done

Mini Kanon 3 is complete when:

- all four capabilities are accessible through one documented interface;
- each capability has a working baseline and an evaluated implementation;
- datasets and preprocessing steps are reproducible;
- the statement-based classifier accepts custom natural-language criteria;
- enrichment outputs pass schema validation;
- segmentation preserves document boundaries and hierarchy;
- the system can process one complete legal-document workflow;
- automated tests pass;
- the service runs locally through Docker;
- results, limitations, and failure modes are documented;
- the repository is understandable to another ML engineer without additional explanation.

---

## Project Positioning

**Mini Kanon 3 is an end-to-end ML engineering project demonstrating how retrieval, universal classification, structured enrichment, and document segmentation can be combined into one practical legal document intelligence system.**
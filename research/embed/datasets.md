# Embed datasets

## Decision

Start with `isaacus/open-australian-legal-qa` v2.0.0. Treat `question` as the query and `source.text` as its positive source passage. Split by source document, preserve all positives before negative mining, and keep MLEB external to training data.

## Alternatives considered

- Generated answers as passages: rejected because this measures answer similarity rather than source retrieval.
- Random row splitting: rejected because passages from one source document could leak across splits.

## Unresolved questions

- Which additional datasets best cover contracts, legislation, and case law?
- Should identical passage text from different source records retain multiple provenance records?

## Experiment required

Measure retrieval performance by jurisdiction and document type after the first zero-shot baseline.

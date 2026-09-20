"""Enrichment layer — derived fields attached to a transaction at ingestion.

Enrichment sits beside scoring, not inside it: it adds reporting figures
to a row and must never be able to stop that row being scored. Phase 5F
adds one enricher, FX conversion into the reporting currency.

See `docs/FX_CONTRACT.md`.
"""

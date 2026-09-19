"""Benchmark tracks that stand outside the canonical schema.

A track reads a source that cannot be adapted into `CanonicalDataset` without
inventing fields it lacks. It still meets the provenance contract — a pinned
manifest digest and a `DatasetProvenance` record — and its runs never reach
the production featureset registry or the served artifacts.
"""

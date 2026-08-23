"""Portable vLLM rewriting pipeline.

A behaviour-preserving port of the JHU two-pass rewrite pipeline
(07_rewrite "wiki" + 09_Distill "distill" + 10_postprocess trim/shuffle) to a single
node with N GPUs and no scheduler.

See docs/SOURCE_INVENTORY.md for what was ported and docs/HANDOFF_REVIEW.md for every
behavioural difference from the source.
"""
__version__ = "1.0.0"

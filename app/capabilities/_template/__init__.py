"""Scaffold for a new capability module.

Copy this directory to `app/capabilities/<feature>/`, rename the `.template`
files to `.py`, fill in the manifest + tools, add a prompts/ and evals/ dir, and
the registry will discover it at boot — no core file is touched. This directory
itself is skipped by discovery because it has no importable `manifest.py`.
See plan §5.6 and Appendix F.
"""

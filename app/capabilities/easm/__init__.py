"""EASM (External Attack Surface Management) capability — the second module.

Proves the chassis: it is a sibling of `reports` with its own manifest + tools +
routing + evals, and the supervisor routes to it with NO core change. Ships dark
(enabled_by_default=False); activate with CAP_EASM_ENABLED=true. See plan §6, §14.
"""

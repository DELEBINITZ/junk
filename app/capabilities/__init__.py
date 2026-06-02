"""Capability MODULES — drop-in features the platform auto-discovers.

Each subpackage with a `manifest.py` exporting `MANIFEST` becomes a registered
capability at boot. `reports` ships in v1; easm/brand/aci are added as sibling
directories with no change to the core. `_template` is a scaffold and is skipped
(it has no importable manifest). See plan §5.
"""

"""Optional, opt-in integrations with third-party libraries.

Nothing under ``swaptrace.integrations`` is imported by ``swaptrace.core`` /
``swaptrace.storage`` / ``swaptrace.pricing`` -- the dependency points one way
only. Each integration has its own optional extra (e.g. ``swaptrace[swapllm]``).
"""

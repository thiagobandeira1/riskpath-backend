"""FastAPI application package for the readmission prediction backend.

See spec.md and plan.md at the project root for design + task context.
The deployed model (`src/inference.py:ReadmissionPredictor`) is imported
by `app.dependencies` and shared across requests via a process-wide singleton.
"""

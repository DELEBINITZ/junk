"""Durable, event-driven ingestion on Temporal (plan §7.1, §13).

Runs ingestion OFF the chat path on an auto-scaled worker fleet. The activity
embeds + upserts a batch of documents into Qdrant for one org; the workflow wraps
it with Temporal's retries/durability. `temporalio` is in the `prod` extra and
imported lazily, so importing this module never requires it.

Start a worker:  .venv/bin/python -m app.core.ingestion.temporal_worker
Trigger:         client.start_workflow(IngestDocumentsWorkflow.run, {...})
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings

logger = logging.getLogger(__name__)

INGEST_WORKFLOW = "IngestDocumentsWorkflow"


def _build():
    """Construct the activity/workflow lazily (needs temporalio)."""

    from temporalio import activity, workflow

    @activity.defn(name="index_documents")
    async def index_documents_activity(payload: dict) -> dict:
        from app.core.ingestion.indexer import QdrantIngestionService

        stats = QdrantIngestionService().index_documents(payload["organization_id"], payload["documents"])
        return {"documents": stats.documents, "chunks": stats.chunks}

    @workflow.defn(name=INGEST_WORKFLOW)
    class IngestDocumentsWorkflow:
        @workflow.run
        async def run(self, payload: dict) -> dict:
            from datetime import timedelta

            return await workflow.execute_activity(
                index_documents_activity,
                payload,
                start_to_close_timeout=timedelta(minutes=10),
            )

    return index_documents_activity, IngestDocumentsWorkflow


async def run_worker() -> None:  # pragma: no cover - needs a Temporal server
    from temporalio.client import Client
    from temporalio.worker import Worker

    activity_fn, workflow_cls = _build()
    client = await Client.connect(settings.temporal_host)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[workflow_cls],
        activities=[activity_fn],
    )
    logger.info("ingestion.worker.start", extra={"task_queue": settings.temporal_task_queue})
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_worker())

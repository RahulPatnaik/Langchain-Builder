from fastapi import BackgroundTasks
from loguru import logger
from typing import Callable, Any

# This file is more of a placeholder when using BackgroundTasks,
# as the tasks are defined and added directly in the endpoint handlers.
# If using Celery/ARQ, this file would define the task functions (@celery.task / @arq.task).

# Example of how a task function might look if defined here (but added in router)
# This function itself isn't directly used by FastAPI's BackgroundTasks mechanism
# but represents the logic that would be passed to `background_tasks.add_task`.

# async def sample_background_task(arg1: str, arg2: int):
#     """Example function intended to be run in the background."""
#     logger.info(f"Background task started with args: {arg1}, {arg2}")
#     try:
#         # Simulate work
#         import asyncio
#         await asyncio.sleep(5) # Simulate I/O bound work
#         result = f"Processed {arg1} with {arg2}"
#         logger.info("Background task finished successfully.")
#         # If needed, store result somewhere (e.g., database, cache)
#         # Be careful with state management in background tasks.
#         return result
#     except Exception as e:
#         logger.error(f"Background task failed: {e}", exc_info=True)
#         # Implement error handling/retry logic if needed


# Helper function to add tasks (optional abstraction)
def add_background_task(
    background_tasks: BackgroundTasks,
    func: Callable,
    *args: Any,
    **kwargs: Any
) -> None:
    """
    Adds a function call to FastAPI's background task queue with logging.

    Args:
        background_tasks: The BackgroundTasks instance from FastAPI.
        func: The function (task) to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    """
    task_name = func.__name__
    logger.info(f"Scheduling background task: {task_name}")
    try:
        background_tasks.add_task(func, *args, **kwargs)
    except Exception as e:
         logger.error(f"Failed to schedule background task {task_name}: {e}", exc_info=True)
         # Decide how to handle scheduling failure (e.g., raise error?)


# Example Usage (in a router):
# from app.services.task_queue import add_background_task, sample_background_task
# @router.post("/submit-job")
# async def submit_job(background_tasks: BackgroundTasks, job_data: dict):
#     add_background_task(
#          background_tasks,
#          sample_background_task,
#          arg1=job_data.get("name"),
#          arg2=job_data.get("value")
#      )
#     return {"message": "Job submitted for background processing"}
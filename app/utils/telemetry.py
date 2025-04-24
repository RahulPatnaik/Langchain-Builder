import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME as RESOURCE_SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter # OTLP GRPC
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter # OTLP HTTP
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
# from opentelemetry.instrumentation.langchain import LangchainInstrumentor # If available and desired

from app.config import settings
from loguru import logger

def init_tracer():
    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT not set. Using ConsoleSpanExporter for tracing.")
        span_exporter = ConsoleSpanExporter()
    else:
        logger.info(f"Initializing OTLP exporter to: {settings.OTEL_EXPORTER_OTLP_ENDPOINT}")
        span_exporter = OTLPSpanExporter(
            endpoint=str(settings.OTEL_EXPORTER_OTLP_ENDPOINT),
            insecure=True # Set to False if using TLS
            # credentials=... # Add credentials if needed
        )

    resource = Resource.create(attributes={
        RESOURCE_SERVICE_NAME: settings.OTEL_SERVICE_NAME,
        "service.version": settings.APP_VERSION,
        # Add other relevant resource attributes: deployment.environment, host.name, etc.
    })

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    logger.info("Tracer provider initialized.")

def instrument_app(app):
    """Instruments the FastAPI app and common libraries."""
    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        logger.warning("Skipping instrumentation as OTLP endpoint is not configured.")
        return

    logger.info("Applying OpenTelemetry instrumentations...")
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    HTTPXClientInstrumentor().instrument() # Instrument httpx globally
    RedisInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    # LangchainInstrumentor().instrument() # Enable if needed

    logger.info("Instrumentations applied.")

# Optional: Helper to get a tracer instance
def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
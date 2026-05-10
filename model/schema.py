from drf_spectacular.utils import extend_schema, OpenApiParameter
from model.serializer import GPTModelQueryRequestSerializer, GPTModelQueryResponseSerializer


model_query_params = extend_schema(
    parameters=[],
    request=GPTModelQueryRequestSerializer,
    responses=GPTModelQueryResponseSerializer,
    tags=["model"],
)

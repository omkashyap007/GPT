from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from model.schema import model_query_params
from model.serializer import GPTModelQueryRequestSerializer, GPTModelQueryResponseSerializer
from trained_model.inference import generate_answer


class GPTModelView(APIView):

    @model_query_params
    def post(self, request, *args, **kwargs):
        serializer = GPTModelQueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        payload = serializer.validated_data
        query = payload["query"]
        answer = generate_answer(query)

        serializer = GPTModelQueryResponseSerializer(data={"response": answer})
        if serializer.is_valid():
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

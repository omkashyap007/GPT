from rest_framework import serializers


class GPTModelQueryRequestSerializer(serializers.Serializer):
    query = serializers.CharField(max_length=1024, required=True)


class GPTModelQueryResponseSerializer(serializers.Serializer):
    response = serializers.CharField(max_length=4096, min_length=0, required=True)

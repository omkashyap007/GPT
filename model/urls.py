from django.urls import path
from model.views import GPTModelView

urlpatterns = [
    path("query/", GPTModelView.as_view(), name="model view"),
]
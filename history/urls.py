from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DetectionHistoryViewSet

router = DefaultRouter()
router.register(r'history', DetectionHistoryViewSet, basename='detection-history')

urlpatterns = [
    path('api/', include(router.urls)),
]
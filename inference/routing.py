from django.urls import path
from .consumers import VideoConsumer
from .optimized_consumers import OptimizedVideoConsumer, UltraLightVideoConsumer

websocket_urlpatterns = [
    path('ws/video/', VideoConsumer.as_asgi()),  # Original
    path('ws/video/optimized/', OptimizedVideoConsumer.as_asgi()),  # Optimized
    path('ws/video/ultralight/', UltraLightVideoConsumer.as_asgi()),  # Ultra-light
]
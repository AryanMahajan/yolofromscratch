from django.urls import path
from .consumers import VideoConsumer
from .optimized_consumers import OptimizedVideoConsumer, OptimizedCoordsVideoConsumer, UltraLightVideoConsumer

websocket_urlpatterns = [
    path('ws/video/', VideoConsumer.as_asgi()),  # Original
    path('ws/video/optimized/', OptimizedVideoConsumer.as_asgi()),  # Optimized with video
    path('ws/video/coords/', OptimizedCoordsVideoConsumer.as_asgi()),  # Coordinates only
    path('ws/video/ultralight/', UltraLightVideoConsumer.as_asgi()),  # Ultra-light
]
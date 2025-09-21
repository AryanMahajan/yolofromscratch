from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from django.db.models import Q
from django_filters.rest_framework import DjangoFilterBackend
from django.utils.dateparse import parse_datetime
from datetime import datetime, timedelta

from .models import DetectionHistory
from .serializers import (
    DetectionHistorySerializer,
    DetectionHistoryListSerializer,
    DetectionHistoryCreateSerializer
)


class DetectionHistoryPagination(PageNumberPagination):
    """Custom pagination for detection history"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class DetectionHistoryViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing user's detection history.
    Only allows users to access their own detection history.
    """
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = DetectionHistoryPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['detection_type']
    
    def get_queryset(self):
        """Return only the current user's detection history"""
        return DetectionHistory.objects.filter(user=self.request.user).order_by('-timestamp')
    
    def get_serializer_class(self):
        """Return appropriate serializer based on action"""
        if self.action == 'list':
            return DetectionHistoryListSerializer
        elif self.action == 'create':
            return DetectionHistoryCreateSerializer
        return DetectionHistorySerializer
    
    def perform_create(self, serializer):
        """Ensure the detection history is created for the current user"""
        serializer.save(user=self.request.user)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent detection history (last 7 days)"""
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_detections = self.get_queryset().filter(timestamp__gte=seven_days_ago)
        
        serializer = DetectionHistoryListSerializer(recent_detections, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get user's detection statistics"""
        user_history = self.get_queryset()
        
        # Total detections
        total_detections = user_history.count()
        
        # Detections by type
        detections_by_type = {}
        for detection_type in ['image', 'video', 'live']:
            count = user_history.filter(detection_type=detection_type).count()
            detections_by_type[detection_type] = count
        
        # Recent activity (last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_count = user_history.filter(timestamp__gte=seven_days_ago).count()
        
        # Total objects detected
        total_objects = sum(history.objects_detected for history in user_history)
        
        # Most common detected classes
        class_counts = {}
        for history in user_history:
            for class_name in history.get_detected_classes():
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Sort by frequency and take top 5
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        stats = {
            'total_detections': total_detections,
            'detections_by_type': detections_by_type,
            'recent_activity': recent_count,
            'total_objects_detected': total_objects,
            'top_detected_classes': top_classes,
            'avg_objects_per_detection': round(total_objects / total_detections, 2) if total_detections > 0 else 0
        }
        
        return Response(stats)
    
    @action(detail=False, methods=['delete'])
    def clear_history(self, request):
        """Clear all detection history for the current user"""
        deleted_count = self.get_queryset().delete()[0]
        return Response(
            {'message': f'Cleared {deleted_count} detection history entries'},
            status=status.HTTP_200_OK
        )
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Search detection history by detected classes or date range"""
        queryset = self.get_queryset()
        
        # Search by detected class
        class_name = request.query_params.get('class', None)
        if class_name:
            # Filter by detection_data containing the class name
            queryset = queryset.filter(
                detection_data__icontains=class_name
            )
        
        # Filter by date range
        start_date = request.query_params.get('start_date', None)
        end_date = request.query_params.get('end_date', None)
        
        if start_date:
            try:
                start_datetime = parse_datetime(start_date)
                if start_datetime:
                    queryset = queryset.filter(timestamp__gte=start_datetime)
            except ValueError:
                pass
        
        if end_date:
            try:
                end_datetime = parse_datetime(end_date)
                if end_datetime:
                    queryset = queryset.filter(timestamp__lte=end_datetime)
            except ValueError:
                pass
        
        # Paginate results
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = DetectionHistoryListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = DetectionHistoryListSerializer(queryset, many=True)
        return Response(serializer.data)
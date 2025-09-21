from django.contrib import admin
from .models import DetectionHistory


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = [
        'user', 
        'detection_type', 
        'objects_detected', 
        'confidence_threshold', 
        'timestamp'
    ]
    list_filter = ['detection_type', 'timestamp', 'confidence_threshold']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['timestamp', 'formatted_timestamp']
    ordering = ['-timestamp']
    
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Detection Details', {
            'fields': (
                'detection_type', 
                'confidence_threshold', 
                'objects_detected',
                'detection_data',
                'image_or_video_ref'
            )
        }),
        ('Timestamp', {
            'fields': ('timestamp', 'formatted_timestamp'),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        """Optimize database queries"""
        return super().get_queryset(request).select_related('user')
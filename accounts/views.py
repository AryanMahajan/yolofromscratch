from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from dj_rest_auth.registration.views import SocialLoginView
from decouple import config
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User

from dj_rest_auth.registration.views import VerifyEmailView
from django.shortcuts import redirect
from django.http import Http404


class CustomConfirmEmailView(VerifyEmailView):
    def get(self, request, *args, **kwargs):
        try:
            self.object = self.get_object()
            self.object.confirm(self.request)
            return redirect(f"{config('FRONTEND_BASE_URL')}/email-confirmed")
        except Http404:
            return redirect(f"{config('FRONTEND_BASE_URL')}/email-confirm-failed")


class GoogleLogin(SocialLoginView):
    adapter_class = GoogleOAuth2Adapter
    client_class = OAuth2Client
    callback_url = f"{config('FRONTEND_BASE_URL')}/google-callback"


@api_view(['GET', 'PUT', 'PATCH'])
@permission_classes([permissions.IsAuthenticated])
def user_me(request):
    """
    Get or update current user's information
    """
    user = request.user
    
    if request.method == 'GET':
        return Response({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'date_joined': user.date_joined,
            'last_login': user.last_login,
            'is_active': user.is_active,
        })
    
    elif request.method in ['PUT', 'PATCH']:
        # Update user information
        data = request.data
        
        # Only allow updating certain fields
        allowed_fields = ['first_name', 'last_name', 'email']
        updated_fields = []
        
        for field in allowed_fields:
            if field in data:
                setattr(user, field, data[field])
                updated_fields.append(field)
        
        if updated_fields:
            user.save()
            return Response({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'date_joined': user.date_joined,
                'last_login': user.last_login,
                'is_active': user.is_active,
                'updated_fields': updated_fields
            })
        else:
            return Response(
                {'error': 'No valid fields provided for update'},
                status=status.HTTP_400_BAD_REQUEST
            )
from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from dj_rest_auth.registration.views import SocialLoginView
from decouple import config

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

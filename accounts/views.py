from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from dj_rest_auth.registration.views import SocialLoginView

from dj_rest_auth.registration.views import VerifyEmailView
from django.shortcuts import redirect
from django.http import Http404

class CustomConfirmEmailView(VerifyEmailView):
    def get(self, request, *args, **kwargs):
        try:
            self.object = self.get_object()
            self.object.confirm(self.request)
            return redirect("http://localhost:5173/email-confirmed")
        except Http404:
            return redirect("http://localhost:5173/email-confirm-failed")



class GoogleLogin(SocialLoginView):
    adapter_class = GoogleOAuth2Adapter
    client_class = OAuth2Client
    callback_url = 'http://localhost:5173/google-callback'

from django.urls import path, include
from .views import GoogleLogin, CustomConfirmEmailView, user_me
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/user/me/', user_me, name='user_me'),
    path('google/', GoogleLogin.as_view(), name='google_login'),
    path(
        "api/auth/registration/account-confirm-email/<str:key>/",
        CustomConfirmEmailView.as_view(),
        name="account_confirm_email",
    ),
]
# Backend Authentication Setup

This document details the setup of the authentication system within the Django backend.

### 1. Installed Packages

The following Python packages were installed to support authentication and API functionality:

- `djangorestframework`
- `djangorestframework-simplejwt`
- `django-allauth`
- `dj-rest-auth==5.0.2`
- `django-cors-headers`
- `google-auth-oauthlib`
- `google-api-python-client`
- `drf-yasg` (for API documentation)
- `cryptography`
- `python-decouple` (for environment variables)

### 2. Core Configuration (`core/settings.py`)

- **`.env` File:** An `.env` file was added to the `yolofromscratch` root to securely store `SECRET_KEY`, `DATABASE_URL`, `GOOGLE_CLIENT_ID`, and `GOOGLE_CLIENT_SECRET`.
- **INSTALLED_APPS:** Added `rest_framework`, `rest_framework.authtoken`, `corsheaders`, `allauth`, `allauth.account`, `allauth.socialaccount`, `allauth.socialaccount.providers.google`, `dj_rest_auth`, `dj_rest_auth.registration`, `drf_yasg`, and the new `accounts` app.
- **MIDDLEWARE:** `corsheaders.middleware.CorsMiddleware` and `allauth.account.middleware.AccountMiddleware` were added.
- **REST_FRAMEWORK:** Configured to use `JWTAuthentication`.
- **CORS:** Configured to allow requests from the frontend at `http://localhost:5173`.
- **AUTHENTICATION_BACKENDS:** Set up to use `django-allauth`'s authentication backend alongside the default.
- **`django-allauth` Settings:** Configured for email authentication, mandatory email verification, and Google as a social account provider.

### 3. `accounts` App

A new Django app, `accounts`, was created to handle custom authentication logic.

- **`views.py`:** Contains the `GoogleLogin` view which handles the server-side of the Google OAuth2 flow.
- **`urls.py`:** Defines the `/google/` endpoint for the Google login view.
- **`serializers.py`:** A `CustomRegisterSerializer` was created to include `first_name` and `last_name` during user registration.

### 4. URL Structure (`core/urls.py`)

The main URL configuration was updated to include:

- **`/accounts/`:** Includes the URLs from the `accounts` app (e.g., `/accounts/google/`).
- **`/api/auth/`:** Includes the primary URLs from `dj-rest-auth` for login, logout, etc.
- **`/api/auth/registration/`:** Includes URLs from `dj-rest-auth` for registration.
- **`/swagger/` and `/redoc/`:** Endpoints for API documentation.

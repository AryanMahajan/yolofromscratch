# API Documentation

This document provides a detailed overview of the APIs in the YOLO from scratch backend.

## Authentication APIs

The authentication system is built using `dj-rest-auth` and `rest_framework_simplejwt`.

### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/accounts/api/token/` | Obtain a new JWT token pair (access and refresh). |
| POST | `/accounts/api/token/refresh/` | Refresh an expired access token using a refresh token. |
| POST | `/accounts/google/` | Handle Google social login. |
| GET | `/accounts/api/auth/registration/account-confirm-email/<str:key>/` | Confirm user's email address. |
| POST | `/api/auth/login/` | User login with email and password. |
| POST | `/api/auth/logout/` | User logout. |
| POST | `/api/auth/password/reset/` | Request a password reset email. |
| POST | `/api/auth/password/reset/confirm/` | Confirm a password reset. |
| GET | `/api/auth/user/` | Get user details. |
| PUT, PATCH | `/api/auth/user/` | Update user details. |
| POST | `/api/auth/registration/` | User registration. |
| POST | `/api/auth/registration/verify-email/` | Verify user's email address. |
| POST | `/api/auth/registration/resend-email/` | Resend email verification. |

### `dj-rest-auth`

For more detailed information about the `dj-rest-auth` endpoints, please refer to the official documentation: [https://dj-rest-auth.readthedocs.io/en/latest/](https://dj-rest-auth.readthedocs.io/en/latest/)

## Inference APIs

### Real-time Video Streaming

The application uses WebSockets for real-time video streaming and object detection.

**Endpoint:** `ws/video/`

**Description:**

This WebSocket endpoint receives a video stream from the client, processes each frame with the YOLO model for object detection, and sends the annotated frames back to the client.

**Message Format:**

*   **Client to Server:**

    The client sends a JSON object with a `frame` key containing a base64 encoded image data URI.

    ```json
    {
        "frame": "data:image/jpeg;base64,"
    }
    ```

*   **Server to Client:**

    The server sends back a JSON object with a `frame` key containing the base64 encoded and annotated image data URI.

    ```json
    {
        "frame": "data:image/jpeg;base64,"
    }
    ```


## API Documentation UI

The project uses `drf-yasg` to generate Swagger and Redoc API documentation.

*   **Swagger UI:** `/swagger/`
*   **Redoc UI:** `/redoc/`

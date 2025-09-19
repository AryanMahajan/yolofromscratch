# YOLO from Scratch and Web App

**Date Created:** 2025-09-19
**Last Updated:** 2025-09-19

## Project Goals

- Implement a YOLO model (v5 or v7) from scratch.
- Train a pre-trained YOLO model on a custom dataset.
- Develop a web application to showcase the models.

## Technologies

- **Backend:** Django
- **Database:** PostgreSQL
- **Deep Learning Framework:** PyTorch or TensorFlow
- **Frontend:** React (TSX and Tailwind)
- **Deployment** GCP (GCR and Jenkins/Github Actions)

---

# Development Plan

This document outlines the steps to build the YOLO models and the web application.

## 1. Project Setup

- [X] Create `dev_docs` directory.
- [X] Create `development_plan.md`.
- [X] Create `model` directory for YOLO model from scratch.
- [X] Create `pretrained_model` directory for transfer learning.

## 2. YOLO Model from Scratch (`/model`)

- **Model Selection:** Decide between YOLOv5 and YOLOv7. Research and document the pros and cons of each for this project.
- **Architecture:** Implement the model architecture from scratch using a deep learning framework like PyTorch or TensorFlow.
    - Backbone (e.g., CSPDarknet)
    - Neck (e.g., PANet)
    - Head (YOLO layer)
- **Training:**
    - Prepare a standard dataset (e.g., COCO).
    - Implement the training loop, including loss functions (e.g., CIoU loss).
    - Train the model.
- **Evaluation:** Evaluate the model's performance on a validation set.

## 3. Pre-trained YOLO Model (`/pretrained_model`)

- **Model Selection:** Choose a pre-trained YOLO model (e.g., from Ultralytics).
- **Custom Dataset:**
    - Gather and annotate a custom dataset.
    - Prepare the dataset in the required YOLO format.
- **Training:**
    - Set up the training configuration file.
    - Train the model on the custom dataset (transfer learning).
- **Evaluation:** Evaluate the trained model on a test set.

## 4. Web Application

- **Backend (Django):**
    - Create Django apps for each part of the project (e.g., `yolo_scratch`, `yolo_pretrained`).
    - Develop API endpoints to:
        - Upload an image or video for inference.
        - Run inference using the "from scratch" model.
        - Run inference using the pre-trained model.
        - Display the results (bounding boxes, class labels).
- **Frontend:**
    - Design the user interface.
    - Develop the frontend using React with TypeScript and Tailwind CSS.
    - The frontend should allow users to upload an image/video and see the output from the models.

## 5. Documentation and Tracking

- Keep this document updated with progress.
- Document the code and experiments.

## 6. Deployment

- **Cloud Provider:** Google Cloud Platform (GCP).
- **Containerization:** Use Docker to containerize the Django application and the ML models.
- **Container Registry:** Use Google Container Registry (GCR) to store the Docker images.
- **CI/CD:** Set up a CI/CD pipeline using Jenkins/Github Actions to automate testing and deployment to GCP.
- **Orchestration:** For more complex deployments, consider Google Kubernetes Engine (GKE). // (Optional)

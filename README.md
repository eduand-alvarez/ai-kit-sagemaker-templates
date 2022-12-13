# ai-kit-sagemaker-templates
A repository that supports tutorials on building sagemaker pipelines with Intel AI Analytics Toolkit (AI Kit) components. 

## General Project Structure

Each template folder in the repo is named after the AI Kit component that it was tailored for. At the moment, we only have as single template for XGBoost and Daal4py optimizations on Xeon CPUs.

```
Template Folder Example

├───0_xgboost-daal4py-container
├───1_lambda-container
└───2_pipeline-code
    ├───0_model-development
    │   ├───pipelines
    │   │   └───customer_churn
    │   └───tests
    └───1_model-deployment
        └───tests
```

## Decription of Components

### 1_ai-kit-component-container
This folder contains all of the custom model container components. This is the most important folder in the repo because it was the main motivation behind this intiative. SageMaker typically expects you to use their pre-package images for their pipelines but the containers in this repo will allows you to incorporate hardware accelerated libraries into your trianing and inference pipeline components. 
    
### 2_lambda-container
This folder contains all of the components of a custom AWS Lambda function. You will only need this if you are looking to use this particular solution architecture. The architecture will allow you to avoid setting up dedicated servers to monitor incoming requests and execute the code. There are many benefits of this like only paying for the compute every time the lambda function is triggered by incoming requests, instead of a dedicated server.
    
### 3_pipeline-code
This folder contains the edited code that typically comes with the "MLOps template for model building, training, and deployment" available as part of the SageMaker service. The files in this template can evolve as AWS updates their APIs but the general idea will remain the same. In the accompanying tutorials for each AI Kit component, you will find instructions for what files to replace/edit. 


### List of Associated Medium Articles
- A Guide to Implementing Customized Training and Inference in AWS SageMaker
- Custom SageMaker Model Images for Accelerated Machine Learning Libraries
- Building AWS Lambda From ECR Image for Machine Learning Inference Endpoint

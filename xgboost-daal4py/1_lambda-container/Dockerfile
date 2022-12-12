# download base image
FROM public.ecr.aws/lambda/python:3.8

# copy our lambda handler script
COPY app.py ${LAMBDA_TASK_ROOT}


# install our dependencies
COPY requirements.txt .
RUN pip3 --no-cache-dir install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# execute function for sagemaker endpoint prediction
RUN echo Utilizing SageMaker Endpoint
CMD [ "app.sagemaker_endpoint" ]
## MLOps for SageMaker Endpoint Deployment

This is a sample code repository for demonstrating how you can organize your code for deploying an realtime inference Endpoint infrastructure.

## Layout of the SageMaker ModelBuild Project Template

`buildspec.yml`
 - this file is used by the CodePipeline's Build stage to build a CloudFormation template.

`build.py`
 - this python file contains code to get the latest approve package arn and exports staging and configuration files. This is invoked from the Build stage.

`endpoint-config-template.yml`
 - this CloudFormation template file is packaged by the build step in the CodePipeline and is deployed in different stages.

`staging-config.json`
 - this configuration file is used to customize `staging` stage in the pipeline. You can configure the instance type, instance count here.

`prod-config.json`
 - this configuration file is used to customize `prod` stage in the pipeline. You can configure the instance type, instance count here.

`test\buildspec.yml`
  - this file is used by the CodePipeline's `staging` stage to run the test code of the following python file

`test\test.py`
  - this python file contains code to describe and invoke the staging endpoint. You can customize to add more tests here.

## Using AWS Bedrock

If you want to use AWS Bedrock you should install bedrock-python-sdk

```bash
pip install https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip
```

setup your AWS [creds](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) 


```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=.....
export AWS_DEFAULT_REGION=us-west-2  # bedrock region
```

```py hl_lines="17"
--8<-- "examples/bedrock.py"
```
## Using AWS Bedrock

If you want to use AWS Bedrock you should install bedrock-python-sdk

```bash
pip install 'pydantic-prompter[bedrock]'
```

setup your AWS [creds](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) 

```py hl_lines="17"
--8<-- "examples/bedrock.py"
```

## Using Cohere

If you want to use AWS Bedrock you should install bedrock-python-sdk

```bash
pip install 'pydantic-prompter[cohere]'
```

setup your AWS [creds](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) 

```py hl_lines="17"
--8<-- "examples/cohere.py"
```
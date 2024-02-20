class NonRetryable(Exception):
    pass


class Retryable(Exception):
    pass


class OpenAiAuthenticationError(NonRetryable):
    pass


class BedRockAuthenticationError(NonRetryable):
    pass


class FailedToParsePromptError(NonRetryable):
    pass


class BadRoleError(NonRetryable):
    pass


class FailedToCastLLMResult(Retryable):
    pass


class CohereAuthenticationError(NonRetryable):
    pass


class ArgumentError(NonRetryable):
    pass

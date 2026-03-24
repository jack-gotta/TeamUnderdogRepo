def get_ailab_endpoint():
    import os
    if 'AILAB_ENDPOINT' in os.environ:
        return os.environ['AILAB_ENDPOINT']    
    return 'https://ct-enterprisechat-api.azure-api.net/'


def authenticate_ailab_interactive():
    """Authenticate interactively and return a reusable bearer token.

    This is intended for local development flows such as notebooks, where a user
    wants the Python pipeline to handle sign-in directly instead of relying on
    `azd auth login` ahead of time.
    """
    from azure.identity import DeviceCodeCredential, InteractiveBrowserCredential

    scope = "api://ailab/Model.Access"
    try:
        credential = InteractiveBrowserCredential()
        token = credential.get_token(scope)
        return {
            "token": token.token,
            "expires_on": token.expires_on,
            "auth_method": "interactive_browser",
        }
    except Exception:
        credential = DeviceCodeCredential()
        token = credential.get_token(scope)
        return {
            "token": token.token,
            "expires_on": token.expires_on,
            "auth_method": "device_code",
        }


def get_ailab_auth_status() -> dict[str, str | bool | None]:
    """Return a lightweight view of whether server-side auth is available.

    This function intentionally avoids any interactive prompts. It inspects the
    current environment and then attempts a non-interactive token acquisition
    chain. It is safe to use in API status endpoints.
    """
    import os

    scope = "api://ailab/Model.Access"
    if os.environ.get("AILAB_BEARER_TOKEN"):
        return {
            "authenticated": True,
            "auth_source": "env_token",
            "scope": scope,
        }

    try:
        from azure.identity import (
            AzureCliCredential,
            AzureDeveloperCliCredential,
            ChainedTokenCredential,
            EnvironmentCredential,
            ManagedIdentityCredential,
            SharedTokenCacheCredential,
        )

        credential = ChainedTokenCredential(
            EnvironmentCredential(),
            ManagedIdentityCredential(),
            SharedTokenCacheCredential(),
            AzureCliCredential(),
            AzureDeveloperCliCredential(),
        )
        credential.get_token(scope)
        return {
            "authenticated": True,
            "auth_source": "credential_chain",
            "scope": scope,
        }
    except Exception as exc:
        return {
            "authenticated": False,
            "auth_source": None,
            "scope": scope,
            "error": str(exc),
        }

def get_ailab_bearer_token_provider():
    """
    Retrieves a bearer token provider for AI Lab model access using Azure credentials.
    
    The Azure credentials are loaded using the [DefaultAzureCredential](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) class from the [azure-identity](https://pypi.org/project/azure-identity/) package. See the documentation for the sequence of authentication methods used to try and obtain credentials.
    
    For local use, the recommended path is via SSO using the `azd auth login --scope api://ailab/Model.Access` command. This will store the credentials in the local cache and allow the DefaultAzureCredential to retrieve them.

    Returns:
        Callable: A token provider function that can be used to obtain bearer tokens.

    Example:
        token_provider = get_ailab_bearer_token_provider()
        token = token_provider()
    """
    import os
    static_token = os.environ.get("AILAB_BEARER_TOKEN")
    if static_token:
        return lambda: static_token

    from azure.identity import (
        AzureCliCredential,
        AzureDeveloperCliCredential,
        ChainedTokenCredential,
        EnvironmentCredential,
        InteractiveBrowserCredential,
        ManagedIdentityCredential,
        SharedTokenCacheCredential,
        get_bearer_token_provider as _get_bearer_token_provider,
    )

    credential = ChainedTokenCredential(
        EnvironmentCredential(),
        ManagedIdentityCredential(),
        SharedTokenCacheCredential(),
        AzureCliCredential(),
        AzureDeveloperCliCredential(),
        InteractiveBrowserCredential(),
    )
    token_provider = _get_bearer_token_provider(credential, "api://ailab/Model.Access")
    return token_provider
"""Pytest-style unit tests for PromptsClient.

These tests were converted from unittest to pytest. They keep the same
behaviour but use pytest fixtures, parametrization and asyncio support.
"""

from unittest.mock import MagicMock, patch

import pytest

from basalt._internal.http import HTTPResponse
from basalt.prompts.client import PromptsClient
from basalt.prompts.models import (
    DescribePromptResponse,
    PromptListResponse,
    PromptModel,
    PromptModelParameters,
    PromptResponse,
    PublishPromptResponse,
)
from basalt.types.exceptions import BadRequestError, NotFoundError, UnauthorizedError


@pytest.fixture
def common_client():
    api_key = "test-api-key"
    cache = MagicMock()
    cache.get.return_value = None
    fallback_cache = MagicMock()
    fallback_cache.get.return_value = None

    client = PromptsClient(
        api_key=api_key,
        cache=cache,
        fallback_cache=fallback_cache,
    )

    mock_prompt_model = PromptModel(
        provider="open-ai",
        model="gpt-4o",
        version="latest",
        parameters=PromptModelParameters(
            temperature=0.5,
            max_length=100,
            response_format="json_object",
            top_p=0.5,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            json_object=None,
        ),
    )

    mock_prompt_response = PromptResponse(
        text="Hello {{name}}",
        slug="test-slug",
        version="1.0.0",
        tag="prod",
        model=mock_prompt_model,
        system_text="You are a helpful assistant",
    )

    return {
        "client": client,
        "cache": cache,
        "fallback_cache": fallback_cache,
        "mock_prompt_model": mock_prompt_model,
        "mock_prompt_response": mock_prompt_response,
    }


def make_response(payload: dict | None, status: int = 200) -> HTTPResponse:
    return HTTPResponse(status_code=status, data=payload, headers={})


def test_get_sync_success(common_client):
    client: PromptsClient = common_client["client"]
    cache = common_client["cache"]
    fallback_cache = common_client["fallback_cache"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "text": "Hello {{name}}",
                    "slug": "test-slug",
                    "version": "1.0.0",
                    "tag": "prod",
                    "systemText": "You are a helpful assistant",
                    "model": {
                        "provider": "open-ai",
                        "client": "open-ai",
                        "model": "gpt-4o",
                        "version": "latest",
                        "parameters": {
                            "temperature": 0.5,
                            "maxLength": 100,
                            "responseFormat": "json_object",
                            "topP": 0.5,
                        },
                    },
                },
            }
        )

        prompt = client.get_sync("test-slug", version="1.0.0", tag="prod")

        # Verify API was called
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert "/prompts/test-slug" in call_kwargs["url"]
        assert call_kwargs["params"]["version"] == "1.0.0"
        assert call_kwargs["params"]["tag"] == "prod"

        # Verify prompt object (wrapped in PromptContextManager)
        from basalt.prompts.models import PromptContextManager

        assert isinstance(prompt, PromptContextManager)
        # Verify attributes are forwarded correctly
        assert prompt.slug == "test-slug"
        assert prompt.version == "1.0.0"
        assert prompt.tag == "prod"

        # Verify caching
        cache.put.assert_called_once()
        fallback_cache.put.assert_called_once()


def test_get_sync_with_variables(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "text": "Hello {{name}}",
                    "slug": "test-slug",
                    "version": "1.0.0",
                    "tag": "prod",
                    "systemText": "You are {{role}}",
                    "model": {
                        "provider": "open-ai",
                        "client": "open-ai",
                        "model": "gpt-4o",
                        "version": "latest",
                        "parameters": {
                            "temperature": 0.5,
                            "maxLength": 100,
                            "responseFormat": "json_object",
                        },
                    },
                },
            }
        )

        variables = {"name": "World", "role": "helpful"}
        prompt = client.get_sync("test-slug", variables=variables)

        # Verify variable substitution
        assert prompt.text == "Hello World"
        assert prompt.system_text == "You are helpful"
        assert prompt.raw_text == "Hello {{name}}"


def test_get_sync_cache_hit(common_client):
    client: PromptsClient = common_client["client"]
    cache = common_client["cache"]
    mock_prompt_response = common_client["mock_prompt_response"]

    cache.get.return_value = mock_prompt_response

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        prompt = client.get_sync("test-slug")

        # Verify API was NOT called
        mock_fetch.assert_not_called()

        # Verify prompt was returned from cache
        assert prompt.slug == "test-slug"


def test_get_sync_cache_disabled(common_client):
    client: PromptsClient = common_client["client"]
    cache = common_client["cache"]
    fallback_cache = common_client["fallback_cache"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "text": "Hello",
                    "slug": "test-slug",
                    "version": "1.0.0",
                    "tag": "prod",
                    "systemText": "",
                    "model": {
                        "provider": "open-ai",
                        "client": "open-ai",
                        "model": "gpt-4o",
                        "version": "latest",
                        "parameters": {
                            "temperature": 0.5,
                            "maxLength": 100,
                            "responseFormat": "json_object",
                        },
                    },
                },
            }
        )

        # Set cache to have a value
        cache.get.return_value = common_client["mock_prompt_response"]

        client.get_sync("test-slug", cache_enabled=False)

        # Verify API WAS called despite cache
        mock_fetch.assert_called_once()

        # Verify caching was skipped
        cache.put.assert_not_called()
        fallback_cache.put.assert_not_called()


def test_get_sync_fallback_cache(common_client):
    client: PromptsClient = common_client["client"]
    fallback_cache = common_client["fallback_cache"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        # Mock API error
        mock_fetch.side_effect = NotFoundError("Prompt not found")

        # Set fallback cache to have a value
        fallback_cache.get.return_value = common_client["mock_prompt_response"]

        prompt = client.get_sync("test-slug")

        # Verify fallback cache was used
        fallback_cache.get.assert_called_once()
        assert prompt.slug == "test-slug"


def test_get_sync_error_no_fallback(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.side_effect = NotFoundError("Prompt not found")

        with pytest.raises(NotFoundError):
            client.get_sync("test-slug")


def test_describe_sync_success(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "slug": "test-slug",
                    "status": "live",
                    "name": "Test Prompt",
                    "description": "A test prompt",
                    "availableVersions": ["1.0.0", "1.1.0"],
                    "availableTags": ["latest", "production", "staging"],
                    "variables": [
                        {
                            "label": "Name",
                            "type": "string",
                            "description": "This is a description of the variable",
                        }
                    ],
                },
            }
        )

        response = client.describe_sync("test-slug", version="1.0.0")

        assert isinstance(response, DescribePromptResponse)
        assert response.slug == "test-slug"
        assert response.status == "live"
        assert response.name == "Test Prompt"
        assert len(response.available_versions) == 2
        assert len(response.available_tags) == 3


def test_describe_sync_error(common_client):
    client = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.side_effect = UnauthorizedError("Invalid API key")

        with pytest.raises(UnauthorizedError):
            client.describe_sync("test-slug")


def test_list_sync_success(common_client):
    client = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "prompts": [
                    {
                        "slug": "prompt-1",
                        "status": "live",
                        "name": "Prompt 1",
                        "description": "First prompt",
                        "availableVersions": ["1.0.0"],
                        "availableTags": ["latest"],
                    },
                    {
                        "slug": "prompt-2",
                        "status": "live",
                        "name": "Prompt 2",
                        "description": "Second prompt",
                        "availableVersions": ["2.0.0"],
                        "availableTags": ["production"],
                    },
                ]
            }
        )

        prompts = client.list_sync()

        assert len(prompts) == 2
        assert isinstance(prompts[0], PromptListResponse)
        assert prompts[0].slug == "prompt-1"
        assert prompts[1].slug == "prompt-2"


def test_list_sync_with_feature_slug(common_client):
    client = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response({"prompts": []})

        client.list_sync(feature_slug="my-feature")

        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["params"]["featureSlug"] == "my-feature"


def test_list_sync_error(common_client):
    client = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.side_effect = BadRequestError("Invalid request")

        with pytest.raises(BadRequestError):
            client.list_sync()


@pytest.mark.parametrize(
    "slug,version,tag",
    [
        ("test-slug", "1.0.0", "prod"),
        ("test-slug", "1.0.0", None),
        ("test-slug", None, "prod"),
        ("test-slug", None, None),
    ],
)
def test_get_sync_parameter_combinations(common_client, slug, version, tag):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "text": "Test",
                    "slug": slug,
                    "version": version or "latest",
                    "tag": tag or "default",
                    "systemText": "",
                    "model": {
                        "provider": "open-ai",
                        "client": "open-ai",
                        "model": "gpt-4o",
                        "version": "latest",
                        "parameters": {
                            "temperature": 0.5,
                            "maxLength": 100,
                            "responseFormat": "json_object",
                        },
                    },
                },
            }
        )

        client.get_sync(slug, version=version, tag=tag)

        call_kwargs = mock_fetch.call_args[1]
        params = call_kwargs["params"]

        if version:
            assert params.get("version") == version
        if tag:
            assert params.get("tag") == tag


@pytest.mark.asyncio
async def test_get_async_success(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "text": "Hello {{name}}",
                    "slug": "test-slug",
                    "version": "1.0.0",
                    "tag": "prod",
                    "systemText": "You are a helpful assistant",
                    "model": {
                        "provider": "open-ai",
                        "client": "open-ai",
                        "model": "gpt-4o",
                        "version": "latest",
                        "parameters": {
                            "temperature": 0.5,
                            "maxLength": 100,
                            "responseFormat": "json_object",
                            "topP": 0.5,
                        },
                    },
                },
            }
        )

        prompt = await client.get("test-slug", version="1.0.0")

        mock_fetch.assert_called_once()
        # Verify prompt object (wrapped in AsyncPromptContextManager)
        from basalt.prompts.models import AsyncPromptContextManager

        assert isinstance(prompt, AsyncPromptContextManager)
        # Verify attributes are forwarded correctly
        assert prompt.slug == "test-slug"


@pytest.mark.asyncio
async def test_get_async_cache_hit(common_client):
    client: PromptsClient = common_client["client"]
    cache = common_client["cache"]
    mock_prompt_response = common_client["mock_prompt_response"]

    cache.get.return_value = mock_prompt_response

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        prompt = await client.get("test-slug")

        mock_fetch.assert_not_called()
        assert prompt.slug == "test-slug"


@pytest.mark.asyncio
async def test_get_async_fallback_cache(common_client):
    client = common_client["client"]
    fallback_cache = common_client["fallback_cache"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.side_effect = NotFoundError("Prompt not found")
        fallback_cache.get.return_value = common_client["mock_prompt_response"]

        prompt = await client.get("test-slug")

        assert prompt.slug == "test-slug"


@pytest.mark.asyncio
async def test_describe_async_success(common_client):
    client = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "slug": "test-slug",
                    "status": "live",
                    "name": "Test Prompt",
                    "description": "A test prompt",
                    "availableVersions": ["1.0.0"],
                    "availableTags": ["latest", "production", "staging"],
                    "variables": [],
                },
            }
        )

        response = await client.describe("test-slug")

        assert isinstance(response, DescribePromptResponse)
        assert response.slug == "test-slug"


@pytest.mark.asyncio
async def test_list_async_success(common_client):
    client = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "prompts": [
                    {
                        "slug": "prompt-1",
                        "status": "live",
                        "name": "Prompt 1",
                        "description": "First prompt",
                        "availableVersions": ["1.0.0"],
                        "availableTags": ["latest"],
                    },
                ]
            }
        )

        prompts = await client.list()

        assert len(prompts) == 1
        assert prompts[0].slug == "prompt-1"


@pytest.mark.asyncio
async def test_get_async_with_variables(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "warning": "",
                "prompt": {
                    "text": "Hello {{name}}",
                    "slug": "test-slug",
                    "version": "1.0.0",
                    "tag": "prod",
                    "systemText": "You are {{role}}",
                    "model": {
                        "provider": "open-ai",
                        "client": "open-ai",
                        "model": "gpt-4o",
                        "version": "latest",
                        "parameters": {
                            "temperature": 0.5,
                            "maxLength": 100,
                            "responseFormat": "json_object",
                        },
                    },
                },
            }
        )

        variables = {"name": "Alice", "role": "assistant"}
        prompt = await client.get("test-slug", variables=variables)

        assert prompt.text == "Hello Alice"
        assert prompt.system_text == "You are assistant"


def test_headers_include_api_key():
    client = PromptsClient(
        api_key="test-key",
        cache=MagicMock(),
        fallback_cache=MagicMock(),
    )

    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer test-key"


def test_headers_include_sdk_info():
    client = PromptsClient(
        api_key="test-key",
        cache=MagicMock(),
        fallback_cache=MagicMock(),
    )

    headers = client._get_headers()
    assert "X-BASALT-SDK-VERSION" in headers
    assert "X-BASALT-SDK-TYPE" in headers
    assert headers["X-BASALT-SDK-TYPE"] == "python"


def test_headers_include_content_type():
    client = PromptsClient(
        api_key="test-key",
        cache=MagicMock(),
        fallback_cache=MagicMock(),
    )

    headers = client._get_headers()
    assert headers["Content-Type"] == "application/json"


def test_publish_prompt_sync_success(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "deploymentTag": {
                    "id": "tag-123",
                    "label": "production",
                }
            }
        )

        response = client.publish_sync(
            slug="test-slug",
            new_tag="production",
            version="1.0.0",
        )

        assert isinstance(response, PublishPromptResponse)
        assert response.id == "tag-123"
        assert response.label == "production"

        # Verify API was called correctly
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert "/prompts/test-slug/publish" in call_kwargs["url"]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["body"]["newTag"] == "production"
        assert call_kwargs["body"]["version"] == "1.0.0"


def test_publish_prompt_sync_with_tag(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "deploymentTag": {
                    "id": "tag-456",
                    "label": "staging",
                }
            }
        )

        response = client.publish_sync(
            slug="test-slug",
            new_tag="staging",
            tag="dev",
        )

        assert isinstance(response, PublishPromptResponse)
        assert response.id == "tag-456"
        assert response.label == "staging"

        # Verify API was called correctly
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["body"]["newTag"] == "staging"
        assert call_kwargs["body"]["tag"] == "dev"
        assert "version" not in call_kwargs["body"]


def test_publish_prompt_sync_minimal(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "deploymentTag": {
                    "id": "tag-789",
                    "label": "latest",
                }
            }
        )

        response = client.publish_sync(
            slug="test-slug",
            new_tag="latest",
        )

        assert isinstance(response, PublishPromptResponse)
        assert response.id == "tag-789"
        assert response.label == "latest"

        # Verify only required fields are in body
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["body"]["newTag"] == "latest"
        assert "version" not in call_kwargs["body"]
        assert "tag" not in call_kwargs["body"]


def test_publish_prompt_sync_error(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.side_effect = BadRequestError("Invalid tag name")

        with pytest.raises(BadRequestError):
            client.publish_sync(slug="test-slug", new_tag="invalid tag")


@pytest.mark.asyncio
async def test_publish_prompt_async_success(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "deploymentTag": {
                    "id": "tag-async-123",
                    "label": "production",
                }
            }
        )

        response = await client.publish(
            slug="test-slug",
            new_tag="production",
            version="2.0.0",
        )

        assert isinstance(response, PublishPromptResponse)
        assert response.id == "tag-async-123"
        assert response.label == "production"

        # Verify API was called correctly
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert "/prompts/test-slug/publish" in call_kwargs["url"]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["body"]["newTag"] == "production"
        assert call_kwargs["body"]["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_publish_prompt_async_with_both_version_and_tag(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "deploymentTag": {
                    "id": "tag-both",
                    "label": "release",
                }
            }
        )

        response = await client.publish(
            slug="test-slug",
            new_tag="release",
            version="1.5.0",
            tag="beta",
        )

        assert isinstance(response, PublishPromptResponse)
        assert response.id == "tag-both"
        assert response.label == "release"

        # Verify both version and tag are in body
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["body"]["newTag"] == "release"
        assert call_kwargs["body"]["version"] == "1.5.0"
        assert call_kwargs["body"]["tag"] == "beta"


@pytest.mark.asyncio
async def test_publish_prompt_async_error(common_client):
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.side_effect = UnauthorizedError("Invalid API key")

        with pytest.raises(UnauthorizedError):
            await client.publish(slug="test-slug", new_tag="production")


@pytest.mark.parametrize(
    "slug,new_tag,version,tag",
    [
        ("prompt-1", "prod", "1.0.0", None),
        ("prompt-2", "staging", None, "dev"),
        ("prompt-3", "release", "2.0.0", "beta"),
        ("prompt-4", "latest", None, None),
    ],
)
def test_publish_prompt_sync_parameter_combinations(common_client, slug, new_tag, version, tag):
    client = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "deploymentTag": {
                    "id": "tag-param-test",
                    "label": new_tag,
                }
            }
        )

        client.publish_sync(
            slug=slug,
            new_tag=new_tag,
            version=version,
            tag=tag,
        )

        call_kwargs = mock_fetch.call_args[1]
        body = call_kwargs["body"]

        assert body["newTag"] == new_tag
        if version:
            assert body["version"] == version
        else:
            assert "version" not in body
        if tag:
            assert body["tag"] == tag
        else:
            assert "tag" not in body


def test_get_sync_with_tools(common_client):
    """Test that tools field is properly parsed and included in the Prompt."""
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response({"warning": "", "prompt": {
            "text": "Hello {{name}}",
            "slug": "test-slug",
            "version": "1.0.0",
            "tag": "prod",
            "systemText": "You are a helpful assistant",
            "model": {
                "provider": "openai",
                "model": "gpt-4",
                "version": "1.0",
                "parameters": {
                    "temperature": 0.7,
                    "maxLength": 100,
                    "responseFormat": "text",
                    "tools": {
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "description": "Get the current weather",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "location": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        ],
                        "toolChoice": {"type": "auto"}
                    }
                }
            }
        }})

        prompt = client.get_sync("test-slug")

        assert prompt.tools is not None
        assert len(prompt.tools.tools) == 1
        assert prompt.tools.tools[0]["type"] == "function"
        assert prompt.tools.tools[0]["function"]["name"] == "get_weather"
        assert prompt.tools.tool_choice is not None
        assert prompt.tools.tool_choice["type"] == "auto"


def test_get_sync_without_tools(common_client):
    """Test that prompts without tools field work correctly."""
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response({"warning": "", "prompt": {
            "text": "Hello",
            "slug": "test-slug",
            "version": "1.0.0",
            "tag": "prod",
            "systemText": "You are a helpful assistant",
            "model": {
                "provider": "openai",
                "model": "gpt-4",
                "version": "1.0",
                "parameters": {
                    "temperature": 0.7,
                    "maxLength": 100,
                    "responseFormat": "text",
                }
            }
        }})

        prompt = client.get_sync("test-slug")

        assert prompt.tools is None


@pytest.mark.asyncio
async def test_get_async_with_tools(common_client):
    """Test that tools field is properly parsed in async get."""
    client: PromptsClient = common_client["client"]

    with patch("basalt.prompts.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response({"warning": "", "prompt": {
            "text": "Hello {{name}}",
            "slug": "test-slug",
            "version": "1.0.0",
            "tag": "prod",
            "systemText": "You are a helpful assistant",
            "model": {
                "provider": "openai",
                "model": "gpt-4",
                "version": "1.0",
                "parameters": {
                    "temperature": 0.7,
                    "maxLength": 100,
                    "responseFormat": "text",
                    "tools": {
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "description": "Search the web",
                                }
                            }
                        ]
                    }
                }
            }
        }})

        prompt = await client.get("test-slug")

        assert prompt.tools is not None
        assert len(prompt.tools.tools) == 1
        assert prompt.tools.tools[0]["function"]["name"] == "search"
        assert prompt.tools.tool_choice is None

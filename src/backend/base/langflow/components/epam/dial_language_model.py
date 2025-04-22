from langchain_openai import AzureChatOpenAI

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import MessageTextInput
from langflow.io import DropdownInput, IntInput, SecretStrInput, SliderInput, BoolInput

import aiohttp
import asyncio
from typing import List


class DialLanguageModelComponent(LCModelComponent):
    """Component for connecting to DIAL API using AzureChatOpenAI for language model interactions."""

    display_name: str = "DIAL Language Model"
    description: str = "Connect to DIAL API for language model interactions."
    documentation: str = "https://epam-rail.com/dial_api"
    beta = True
    icon = "brain-circuit"
    name = "DialLanguageModel"

    FALLBACK_MODELS = [
        "gpt-4o",
    ]

    AZURE_OPENAI_API_VERSIONS = [
        "2024-02-01"
    ]

    # Class variables to cache model list
    _models_fetched = False
    _available_models = []

    inputs = [
        *LCModelComponent._base_inputs,
        MessageTextInput(
            name="dial_api_host",
            display_name="DIAL API Host",
            info="The host URL for DIAL API. Example: `https://your-dial-url.com`",
            real_time_refresh=True,
            required=True,
        ),
        SecretStrInput(
            name="dial_api_key",
            display_name="DIAL API Key",
            info="Your DIAL API key for authentication",
            real_time_refresh=True,
            required=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=FALLBACK_MODELS,
            value=FALLBACK_MODELS[0] if len(FALLBACK_MODELS) > 0 else None,
            info="Select the model to use",
            refresh_button=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            info="Controls randomness. Lower values are more deterministic, higher values are more creative.",
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response token by token",
            value=False,
        ),
        DropdownInput(
            name="api_version",
            display_name="API Version",
            options=sorted(AZURE_OPENAI_API_VERSIONS, reverse=True),
            value=next(
                (
                    version
                    for version in sorted(AZURE_OPENAI_API_VERSIONS, reverse=True)
                    if not version.endswith("-preview")
                ),
                AZURE_OPENAI_API_VERSIONS[0],
            ),
        ),
    ]

    def build_model(self) -> LanguageModel:
        """Build and return the AzureChatOpenAI instance configured for DIAL API."""
        azure_endpoint = self.dial_api_host
        azure_deployment = self.model_name
        api_key = self.dial_api_key
        temperature = self.temperature
        stream = self.stream if hasattr(self, "stream") else False
        max_tokens = self.max_tokens
        api_version = self.api_version

        try:
            output = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
                api_key=api_key,
                temperature=temperature,
                streaming=stream,
                max_tokens=max_tokens or None,
            )

            return output

        except Exception as e:
            raise ValueError(f"Could not initialize DIAL API client: {e}")

    def update_build_config(self, build_config, field_value, field_name=None):
        """Update build configuration when input fields change."""
        if field_name in ["dial_api_host", "dial_api_key", "model_name"]:
            try:
                models = self.refresh_models()

                # Update the build_config with the new models
                build_config["model_name"] = {
                    **build_config.get("model_name", {}),
                    "options": models,
                    "value": models[0] if not hasattr(self, "model_name") or self.model_name not in models else self.model_name,
                }

            except Exception:
                import traceback
                traceback.print_exc()

        return build_config

    def refresh_models(self):
        """Fetch and update available models."""
        try:
            # Run the async function using asyncio.run()
            models = asyncio.run(self.fetch_models())

            # Update model options
            for i, input_item in enumerate(self.inputs):
                if input_item.name == "model_name":
                    self.inputs[i].options = models
                    # Update current model if needed
                    if not hasattr(self, "model_name") or self.model_name not in models:
                        self.inputs[i].value = models[0]
                        # Also update the instance attribute
                        self.model_name = models[0]
                    break

            # Store models for future reference
            self.__class__._available_models = models
            self.__class__._models_fetched = True

            return models
        except Exception as e:
            # If refresh fails, return fallback models
            print(f"Error refreshing models: {e}")
            return self.FALLBACK_MODELS

    async def fetch_models(self) -> List[str]:
        """Fetch available models from the DIAL API."""
        # Skip API call if we don't have required credentials
        if not hasattr(self, "dial_api_host") or not self.dial_api_host or not hasattr(self, "dial_api_key") or not self.dial_api_key:
            return self.FALLBACK_MODELS

        api_host = self.dial_api_host.rstrip('/')
        models_url = f"{api_host}/openai/deployments?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.dial_api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        print(f"Failed to fetch models: {response.status}")
                        return self.FALLBACK_MODELS

                    data = await response.json()
                    # Extract model names from the response
                    models = [model.get("id") for model in data.get("data", []) if model.get("id")]

                    # Cache the fetched models
                    if models:
                        self.__class__._available_models = models
                        self.__class__._models_fetched = True
                        return models
                    else:
                        # If no models found, return fallback but don't mark as fetched successfully
                        return self.FALLBACK_MODELS
        except Exception as e:
            # On error, return fallback models
            print(f"Exception during model fetch: {e}")
            return self.FALLBACK_MODELS

---
title: LibreLLM API
seo_title: LibreLLM API | LibreYOLO
description: >-
  LibreLLM sends text and image requests to an OpenAI-compatible endpoint and
  returns native SDK responses.
lead: >-
  LibreLLM sends text and image requests to an OpenAI-compatible endpoint and
  returns native SDK responses.
keywords:
  - LibreLLM API
  - LibreYOLO
last_verified: 1.6.0
snippets:
  request:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLLM, SAMPLE_IMAGE

        # Sends a request to the default hosted model, gpt-5.6-luna.
        # Set OPENAI_API_KEY first, or pass a provider model ID and api_key=.
        llm = LibreLLM()
        response = llm("Describe this image.", image=SAMPLE_IMAGE)
        print(response.output_text)
---

## Install

```bash
pip install "libreyolo[llm]"
```

Configure the endpoint's credentials before sending a request. Model IDs refer to the remote provider; LibreLLM loads no local model weights.

## Request

<code-tabs name="request" />

`LibreLLM(model="gpt-5.6-luna", *, api="responses", base_url=None, api_key=None, prompt=None)` uses Responses by default. Without a known provider prefix, the OpenAI SDK reads the key from `OPENAI_API_KEY`. Select `api="chat.completions"` for a Chat Completions endpoint. Pass an image through `image=`; a string in the positional source argument is text.

Image input accepts paths, HTTP URLs, data URIs, PIL images and NumPy BGR arrays. A constructor `prompt=` applies to each request. Native message objects pass through unchanged.

## Stream and async

`stream=True` returns native streaming events. `await llm.async_call(...)` uses the asynchronous client. Other request arguments forward to the SDK.

## Provider routing

`openai/` and `openrouter/` select the corresponding host and environment-key convention. Unknown prefixes remain part of the model ID. An explicit `base_url` selects a compatible hosted or local endpoint.

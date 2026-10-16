//! Forwards matching HTTP protocols to configured models.
use async_stream::try_stream;
use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::Value;

use crate::{
    model::{ModelConfig, NewLlmCall},
    provider::{custom_headers, CallRecorder, CancelOnDrop, FinishReason},
    Error, Result,
};

use super::{build_response, header_pairs, observe_events, Forwarder, Protocol, Summary};

impl Forwarder {
    pub(in crate::api::byok) async fn native(
        &self,
        protocol: Protocol,
        incoming_headers: &HeaderMap,
        mut body: Value,
        model: &ModelConfig,
    ) -> Result<Response> {
        let call_id = format!("external-api:{}", uuid::Uuid::new_v4());
        let request_url = model.request_url()?;
        let new_call = NewLlmCall {
            call_id: call_id.clone(),
            run_id: call_id.clone(),
            conversation_id: call_id.clone(),
            provider_call_index: 0,
            model_hash: model.model_hash.clone(),
            provider_type: model.provider_type(),
            provider_url: request_url.clone(),
            request_type: model.provider_type(),
            request_url: request_url.clone(),
            model_id: model.model_id.clone(),
            display_name: model.display_name.clone(),
            reasoning_effort: model.reasoning_effort.clone(),
            fast: false,
            message_count: body
                .get("messages")
                .or_else(|| body.get("input"))
                .and_then(Value::as_array)
                .map_or(1, Vec::len),
            tool_count: body
                .get("tools")
                .and_then(Value::as_array)
                .map_or(0, Vec::len),
            detailed: false,
        };
        body["model"] = Value::String(model.model_id.clone());
        let mut upstream_headers = HeaderMap::new();
        for (name, value) in incoming_headers {
            if !matches!(
                name.as_str(),
                "authorization"
                    | "x-api-key"
                    | "host"
                    | "content-length"
                    | "content-type"
                    | "connection"
                    | "transfer-encoding"
                    | "accept-encoding"
                    | "cookie"
                    | "proxy-authorization"
                    | "proxy-authenticate"
                    | "te"
                    | "trailer"
                    | "upgrade"
            ) {
                upstream_headers.append(name, value.clone());
            }
        }
        if model.custom_headers_enabled {
            upstream_headers.extend(custom_headers(&model.custom_headers, &call_id)?);
        }
        match protocol {
            Protocol::Messages => {
                upstream_headers.insert(
                    "x-api-key",
                    model.api_key.parse().map_err(|error| {
                        Error::Config(format!("invalid API key header: {error}"))
                    })?,
                );
                if !upstream_headers.contains_key("anthropic-version") {
                    upstream_headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
                }
            }
            Protocol::Chat | Protocol::Responses => {
                upstream_headers.insert(
                    header::AUTHORIZATION,
                    format!("Bearer {}", model.api_key)
                        .parse()
                        .map_err(|error| {
                            Error::Config(format!("invalid API key header: {error}"))
                        })?,
                );
            }
        }
        let logged_headers = upstream_headers
            .iter()
            .filter(|(name, _)| !crate::model::is_sensitive_header(name.as_str()))
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_owned(), Value::String(value.to_owned())))
            })
            .collect::<serde_json::Map<_, _>>();
        let client = self.clients.provider_client(self.request_timeout).await?;
        let recorder = CallRecorder::start(self.store.clone(), new_call).await?;
        // A client that leaves before the response starts must not leave the call running.
        let cancel_record = recorder.cancel_on_drop();
        recorder
            .request(Value::Object(logged_headers), &body)
            .await?;
        let response = match client
            .post(&request_url)
            .headers(upstream_headers)
            .json(&body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                let error = Error::Http(error);
                recorder.failed(&error).await?;
                return Err(error);
            }
        };
        let status = response.status();
        recorder.response_headers(status.as_u16()).await?;
        let headers = response.headers().clone();
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        if stream && status.is_success() {
            Ok(self.stream_response(
                protocol,
                response,
                recorder,
                cancel_record,
                status,
                &headers,
            ))
        } else {
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    let error = Error::Http(error);
                    recorder.failed(&error).await?;
                    return Err(error);
                }
            };
            recorder.response_chunk(&bytes).await?;
            if status.is_success() {
                if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                    let summary = Summary::from_value(protocol, &value);
                    if let Some(usage) = summary.usage {
                        recorder.usage(usage).await?;
                    }
                    if let Some(message) = summary.failure {
                        recorder.failed(&Error::Provider(message)).await?;
                    } else {
                        recorder
                            .completed(summary.finish.unwrap_or(FinishReason::Stop))
                            .await?;
                    }
                } else {
                    recorder.completed(FinishReason::Stop).await?;
                }
            } else {
                recorder
                    .failed(&Error::Provider(format!("upstream returned {status}")))
                    .await?;
            }
            Ok(build_response(
                status,
                header_pairs(&headers),
                Body::from(bytes),
            ))
        }
    }

    fn stream_response(
        &self,
        protocol: Protocol,
        response: reqwest::Response,
        recorder: CallRecorder,
        cancel_record: CancelOnDrop,
        status: StatusCode,
        headers: &HeaderMap,
    ) -> Response {
        let idle_timeout = self.stream_idle_timeout;
        let output = try_stream! {
            let _cancel = cancel_record;
            let (sender, receiver) = tokio::sync::mpsc::channel::<Bytes>(8);
            let observer = tokio::spawn(observe_events(protocol, receiver));
            let mut upstream = response.bytes_stream();
            loop {
                let next = tokio::time::timeout(idle_timeout, upstream.next()).await;
                let chunk = match next {
                    Ok(Some(Ok(chunk))) => chunk,
                    Ok(None) => break,
                    Ok(Some(Err(error))) => {
                        let error = Error::Http(error);
                        recorder.failed(&error).await?;
                        Err::<Bytes, Error>(error)?
                    }
                    Err(_) => {
                        let error = Error::Provider("native upstream stream idle timeout".into());
                        recorder.failed(&error).await?;
                        Err::<Bytes, Error>(error)?
                    }
                };
                recorder.response_chunk(&chunk).await?;
                let _ = sender.send(chunk.clone()).await;
                yield chunk;
            }
            drop(sender);
            if let Ok(summary) = observer.await {
                if let Some(usage) = summary.usage { recorder.usage(usage).await?; }
                if let Some(message) = summary.failure {
                    recorder.failed(&Error::Provider(message)).await?;
                } else {
                    recorder.completed(summary.finish.unwrap_or(FinishReason::Stop)).await?;
                }
            } else {
                recorder.completed(FinishReason::Stop).await?;
            }
        };
        let output = output.map(|result: Result<Bytes>| result);
        build_response(status, header_pairs(headers), Body::from_stream(output))
    }
}

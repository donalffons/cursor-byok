//! Forwards external API requests without projecting them into canonical messages.
mod native;
mod plugin;

use std::time::Duration;

use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    model::Usage, network::NetworkClients, plugin::PluginRegistry, provider::FinishReason,
    store::Store, Error,
};

use super::protocol::Protocol;

/// Sends verbatim requests to configured models over HTTP, and to plugin models that relay
/// Anthropic Messages themselves.
#[derive(Clone)]
pub struct Forwarder {
    store: Store,
    plugins: PluginRegistry,
    clients: NetworkClients,
    request_timeout: Duration,
    stream_idle_timeout: Duration,
}

impl Forwarder {
    pub fn new(
        store: Store,
        plugins: PluginRegistry,
        clients: NetworkClients,
        request_timeout: Duration,
        stream_idle_timeout: Duration,
    ) -> Self {
        Self {
            store,
            plugins,
            clients,
            request_timeout,
            stream_idle_timeout,
        }
    }
}

/// Builds the client response from upstream headers, dropping hop-by-hop and cookie headers.
fn build_response<'a>(
    status: StatusCode,
    headers: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    body: Body,
) -> Response {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    for (name, value) in headers {
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "content-length"
                | "content-encoding"
                | "transfer-encoding"
                | "connection"
                | "set-cookie"
        ) {
            continue;
        }
        if let (Ok(name), Ok(value)) = (
            axum::http::HeaderName::from_bytes(name.as_bytes()),
            axum::http::HeaderValue::from_bytes(value),
        ) {
            response.headers_mut().append(name, value);
        }
    }
    response
}

fn header_pairs(headers: &HeaderMap) -> impl Iterator<Item = (&str, &[u8])> {
    headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_bytes()))
}

/// Usage, finish reason, and failure observed in a forwarded response.
#[derive(Default)]
struct Summary {
    usage: Option<Usage>,
    finish: Option<FinishReason>,
    failure: Option<String>,
}

impl Summary {
    fn from_value(protocol: Protocol, value: &Value) -> Self {
        let mut summary = Self::default();
        summary.observe(protocol, value);
        summary
    }

    fn observe(&mut self, protocol: Protocol, value: &Value) {
        if matches!(
            value.get("type").and_then(Value::as_str),
            Some("error" | "response.failed")
        ) || value.get("error").is_some_and(|error| !error.is_null())
        {
            self.failure = Some(
                value
                    .pointer("/error/message")
                    .or_else(|| value.pointer("/response/error/message"))
                    .or_else(|| value.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("upstream error event")
                    .to_owned(),
            );
        }
        let usage = match protocol {
            Protocol::Chat => value.get("usage"),
            Protocol::Responses => value
                .pointer("/response/usage")
                .or_else(|| value.get("usage")),
            Protocol::Messages => value
                .pointer("/message/usage")
                .or_else(|| value.get("usage")),
        }
        .filter(|usage| !usage.is_null())
        .map(|usage| crate::provider::native_usage(protocol.provider_type(), usage));
        if let Some(usage) = usage {
            let current = self.usage.get_or_insert_default();
            for (target, incoming) in [
                (&mut current.input_tokens, usage.input_tokens),
                (
                    &mut current.context_input_tokens,
                    usage.context_input_tokens,
                ),
                (&mut current.output_tokens, usage.output_tokens),
                (&mut current.total_tokens, usage.total_tokens),
                (&mut current.cache_read_tokens, usage.cache_read_tokens),
                (&mut current.cache_write_tokens, usage.cache_write_tokens),
                (&mut current.reasoning_tokens, usage.reasoning_tokens),
            ] {
                if incoming.is_some() {
                    *target = incoming;
                }
            }
        }
        let reason = match protocol {
            Protocol::Chat => value
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Protocol::Responses => value
                .pointer("/response/incomplete_details/reason")
                .and_then(Value::as_str),
            Protocol::Messages => value
                .pointer("/delta/stop_reason")
                .or_else(|| value.get("stop_reason"))
                .and_then(Value::as_str),
        };
        self.finish = match reason {
            Some("tool_calls" | "tool_use") => Some(FinishReason::ToolUse),
            Some("length" | "max_tokens" | "max_output_tokens") => Some(FinishReason::Length),
            Some(_) => Some(FinishReason::Stop),
            None => self.finish,
        };
        if matches!(protocol, Protocol::Responses)
            && (value.pointer("/item/type").and_then(Value::as_str) == Some("function_call")
                || value
                    .pointer("/response/output")
                    .and_then(Value::as_array)
                    .is_some_and(|items| {
                        items.iter().any(|item| {
                            item.get("type").and_then(Value::as_str) == Some("function_call")
                        })
                    }))
        {
            self.finish = Some(FinishReason::ToolUse);
        }
    }
}

async fn observe_events(
    protocol: Protocol,
    receiver: tokio::sync::mpsc::Receiver<Bytes>,
) -> Summary {
    let source = ReceiverStream::new(receiver)
        .map(Ok::<_, Error>)
        .eventsource();
    futures_util::pin_mut!(source);
    let mut summary = Summary::default();
    while let Some(event) = source.next().await {
        let Ok(event) = event else {
            break;
        };
        if let Ok(value) = serde_json::from_str::<Value>(&event.data) {
            summary.observe(protocol, &value);
        }
    }
    summary
}

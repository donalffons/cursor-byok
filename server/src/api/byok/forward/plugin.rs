//! Relays external Anthropic Messages requests through plugin providers verbatim.
use async_stream::try_stream;
use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::{
    model::{NewLlmCall, ProviderType},
    plugin::{ForwardItem, PluginInvocationPlan},
    provider::{CallRecorder, FinishReason},
    Error, Result,
};

use super::{build_response, observe_events, Forwarder, Protocol, Summary};

/// Client headers the plugin may pass upstream; the plugin owns every other header.
const CLIENT_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];

impl Forwarder {
    /// Sends the client's body to the plugin unchanged and streams the upstream response back.
    /// The worker records the upstream request and bytes; this relay records usage and outcome.
    pub(in crate::api::byok) async fn plugin_messages(
        &self,
        incoming_headers: &HeaderMap,
        body: Value,
        plan: &PluginInvocationPlan,
    ) -> Result<Response> {
        let call_id = format!("external-api:{}", uuid::Uuid::new_v4());
        let recorder = CallRecorder::start(
            self.store.clone(),
            NewLlmCall {
                call_id: call_id.clone(),
                run_id: call_id.clone(),
                conversation_id: call_id,
                provider_call_index: 0,
                model_hash: plan.model.id.clone(),
                provider_type: ProviderType::Plugin,
                provider_url: plan.request_url.clone(),
                request_type: ProviderType::Plugin,
                request_url: plan.request_url.clone(),
                model_id: plan.model.model_id.clone(),
                display_name: plan.model.display_name.clone(),
                reasoning_effort: None,
                fast: false,
                message_count: body
                    .get("messages")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len),
                tool_count: body
                    .get("tools")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len),
                detailed: false,
            },
        )
        .await?;
        // A client that leaves before the response starts must not leave the call running.
        let cancel_record = recorder.cancel_on_drop();
        let client_headers = CLIENT_HEADERS
            .iter()
            .filter_map(|name| {
                let values = incoming_headers
                    .get_all(*name)
                    .iter()
                    .filter_map(|value| value.to_str().ok())
                    .collect::<Vec<_>>();
                (!values.is_empty()).then(|| (name.to_string(), Value::from(values.join(","))))
            })
            .collect::<serde_json::Map<_, _>>();
        let cancellation = CancellationToken::new();
        let mut items = self.plugins.forward_messages(
            plan.model.id.clone(),
            body,
            Value::Object(client_headers),
            cancellation.clone(),
            recorder.clone(),
        );
        // Cancels the plugin call when the client goes away or the relay ends.
        let cancel_plugin = cancellation.drop_guard();
        let idle_timeout = self.stream_idle_timeout;
        let start = match tokio::time::timeout(idle_timeout, items.next()).await {
            Ok(Some(Ok(ForwardItem::Start { status, headers }))) => StatusCode::from_u16(status)
                .map(|status| (status, headers))
                .map_err(|_| Error::Protocol(format!("plugin relayed HTTP status {status}"))),
            Ok(Some(Ok(ForwardItem::Line(_)))) => Err(Error::Protocol(
                "plugin relayed a response body before its status".into(),
            )),
            Ok(Some(Err(error))) => Err(error),
            Ok(None) => Err(Error::Provider(
                "plugin ended the relay without a response".into(),
            )),
            Err(_) => Err(Error::Provider("plugin relay idle timeout".into())),
        };
        let (status, headers) = match start {
            Ok(start) => start,
            Err(error) => {
                recorder.failed(&error).await?;
                return Err(error);
            }
        };
        let sse = headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("content-type") && value.starts_with("text/event-stream")
        });
        let output = try_stream! {
            let _cancel_plugin = cancel_plugin;
            let _cancel_record = cancel_record;
            let (sender, receiver) = tokio::sync::mpsc::channel::<Bytes>(8);
            let observer = tokio::spawn(observe_events(Protocol::Messages, receiver));
            let mut whole = Vec::new();
            loop {
                let error = match tokio::time::timeout(idle_timeout, items.next()).await {
                    Ok(Some(Ok(ForwardItem::Line(line)))) => {
                        let mut bytes = line.into_bytes();
                        bytes.push(b'\n');
                        let chunk = Bytes::from(bytes);
                        if sse {
                            let _ = sender.send(chunk.clone()).await;
                        } else {
                            whole.extend_from_slice(&chunk);
                        }
                        yield chunk;
                        continue;
                    }
                    Ok(None) => break,
                    Ok(Some(Ok(ForwardItem::Start { .. }))) => {
                        Error::Protocol("plugin relayed a second response".into())
                    }
                    Ok(Some(Err(error))) => error,
                    Err(_) => Error::Provider("plugin relay idle timeout".into()),
                };
                recorder.failed(&error).await?;
                // A relayed upstream error is complete; a failed success stream is truncated.
                if status.is_success() {
                    Err::<Bytes, Error>(error)?;
                }
                return;
            }
            drop(sender);
            let summary = if sse {
                observer.await.unwrap_or_default()
            } else {
                serde_json::from_slice::<Value>(&whole)
                    .map(|value| Summary::from_value(Protocol::Messages, &value))
                    .unwrap_or_default()
            };
            if let Some(usage) = summary.usage {
                recorder.usage(usage).await?;
            }
            match summary.failure {
                Some(message) => recorder.failed(&Error::Provider(message)).await?,
                None => recorder.completed(summary.finish.unwrap_or(FinishReason::Stop)).await?,
            }
        };
        let output = output.map(|result: Result<Bytes>| result);
        Ok(build_response(
            status,
            headers
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_bytes())),
            Body::from_stream(output),
        ))
    }
}

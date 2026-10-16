//! External OpenAI and Anthropic compatible API.
mod forward;
mod models;
mod output;
mod protocol;

use std::sync::Arc;

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::{
    model::ModelInvocation, plugin::PluginRegistry, provider::Provider, store::Store, Error,
};

use self::protocol::Protocol;
pub use forward::Forwarder;

/// Anthropic's Messages request size limit; OpenAI's limits are no smaller.
const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone)]
struct ApiState {
    store: Store,
    plugins: PluginRegistry,
    provider: Arc<dyn Provider>,
    forwarder: Option<Forwarder>,
}

pub fn router(
    store: Store,
    plugins: PluginRegistry,
    provider: Arc<dyn Provider>,
    forwarder: Option<Forwarder>,
) -> Router {
    Router::new()
        .route("/byok/v1/models", get(list_models))
        .route("/byok/v1/chat/completions", post(chat))
        .route("/byok/v1/responses", post(responses))
        .route("/byok/v1/messages", post(messages))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(ApiState {
            store,
            plugins,
            provider,
            forwarder,
        })
}

/// An error in the caller's protocol; model listing uses the OpenAI shape.
fn api_error(
    protocol: Option<Protocol>,
    status: StatusCode,
    message: impl std::fmt::Display,
) -> Response {
    let message = message.to_string();
    let body = match protocol {
        Some(Protocol::Messages) => {
            let kind = match status.as_u16() {
                400 => "invalid_request_error",
                401 => "authentication_error",
                403 => "permission_error",
                404 => "not_found_error",
                413 => "request_too_large",
                429 => "rate_limit_error",
                529 => "overloaded_error",
                _ => "api_error",
            };
            json!({"type":"error","error":{"type":kind,"message":message}})
        }
        Some(Protocol::Chat | Protocol::Responses) | None => {
            let kind = if status.is_client_error() {
                "invalid_request_error"
            } else {
                "server_error"
            };
            json!({"error":{"message":message,"type":kind}})
        }
    };
    (status, Json(body)).into_response()
}

async fn authorized(
    state: &ApiState,
    headers: &HeaderMap,
) -> std::result::Result<(), (StatusCode, String)> {
    let settings = state
        .store
        .external_api_settings()
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    if !settings.enabled {
        return Err((StatusCode::FORBIDDEN, "external API is disabled".into()));
    }
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        })
        .unwrap_or_default();
    if supplied.is_empty() || !constant_time_eq(supplied.as_bytes(), settings.api_key.as_bytes()) {
        return Err((StatusCode::UNAUTHORIZED, "invalid API key".into()));
    }
    Ok(())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let difference = left
        .iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        });
    difference == 0
}

async fn list_models(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err((status, message)) = authorized(&state, &headers).await {
        return api_error(None, status, message);
    }
    match models::list(&state.store, &state.plugins).await {
        Ok(models) => Json(models::response(&models)).into_response(),
        Err(error) => api_error(None, StatusCode::CONFLICT, error),
    }
}

async fn chat(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(state, headers, body, Protocol::Chat).await
}

async fn responses(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(state, headers, body, Protocol::Responses).await
}

async fn messages(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(state, headers, body, Protocol::Messages).await
}

async fn generate(
    state: ApiState,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
    protocol: Protocol,
) -> Response {
    let error = |status, message| api_error(Some(protocol), status, message);
    if let Err((status, message)) = authorized(&state, &headers).await {
        return error(status, message);
    }
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };
    let Some(public_model_id) = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    else {
        return error(StatusCode::BAD_REQUEST, "model is required".into());
    };
    let models = match models::list(&state.store, &state.plugins).await {
        Ok(models) => models,
        Err(failure) => return error(StatusCode::CONFLICT, failure.to_string()),
    };
    let Some(model) = models
        .iter()
        .find(|model| model.public_id == public_model_id)
    else {
        return error(StatusCode::NOT_FOUND, "model not found".into());
    };
    if let Some(forwarder) = &state.forwarder {
        if model
            .internal_id
            .starts_with(crate::plugin::ADAPTER_ID_PREFIX)
        {
            if matches!(protocol, Protocol::Messages) {
                let plan = match state.plugins.plan_model(&model.internal_id).await {
                    Ok(plan) => plan,
                    Err(failure) => return error(StatusCode::NOT_FOUND, failure.to_string()),
                };
                if plan.model.forwards_messages {
                    return match forwarder.plugin_messages(&headers, body, &plan).await {
                        Ok(response) => response,
                        Err(failure) => error(StatusCode::BAD_GATEWAY, failure.to_string()),
                    };
                }
            }
        } else {
            match state.store.model(&model.internal_id).await {
                Ok(Some(config)) if protocol.provider_type() == config.provider_type() => {
                    return match forwarder.native(protocol, &headers, body, &config).await {
                        Ok(response) => response,
                        Err(failure) => error(StatusCode::BAD_GATEWAY, failure.to_string()),
                    };
                }
                Ok(_) => {}
                Err(failure) => {
                    return error(StatusCode::INTERNAL_SERVER_ERROR, failure.to_string())
                }
            }
        }
    }
    let parsed = match protocol::parse(protocol, &body) {
        Ok(parsed) => parsed,
        Err(failure) => return error(StatusCode::BAD_REQUEST, failure.to_string()),
    };
    let id = format!("external-api:{}", uuid::Uuid::new_v4());
    let mut request = parsed.request;
    request.model.model_id = model.internal_id.clone();
    let invocation = ModelInvocation {
        call_id: id.clone(),
        run_id: id.clone(),
        conversation_id: id.clone(),
        provider_call_index: 0,
        request,
    };
    let cancellation = CancellationToken::new();
    let stream = state.provider.stream(invocation, cancellation.clone());
    if parsed.stream {
        output::streamed(protocol, stream, cancellation, id, parsed.public_model_id)
    } else {
        match output::complete(protocol, stream, &id, &parsed.public_model_id).await {
            Ok(response) => response.into_response(),
            Err(Error::Protocol(message)) => error(StatusCode::BAD_REQUEST, message),
            Err(failure) => error(StatusCode::BAD_GATEWAY, failure.to_string()),
        }
    }
}

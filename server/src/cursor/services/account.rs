//! Implements Cursor account information services.
use axum::{
    body::{to_bytes, Body},
    extract::{Extension, State},
    http::{header, HeaderValue, Request, Response},
};
use bytes::{Bytes, BytesMut};
use prost::{encoding, Message};
use serde_json::Value;

use crate::{
    api::cursor::{is_local_model, proxy},
    cursor::{protocol::proto::agent::v1 as agent, transport::TransportRegistry},
    local_app,
    store::Store,
    Result,
};

use super::entitlement::FreeEntitlementCache;

const LOCAL_AUTH_ID: &str = "cursor-local-user";
const LOCAL_EMAIL: &str = "cursor@ai.com";
const LOCAL_ULTRA_PLAN_INCLUDED_CENTS: i32 = 20_000;

#[derive(Clone, PartialEq, Message)]
struct GetEmailResponse {
    #[prost(string, tag = "1")]
    email: String,
    #[prost(int32, tag = "2")]
    sign_up_type: i32,
}

#[derive(Clone, PartialEq, Message)]
struct GetUserMetaResponse {
    #[prost(string, tag = "1")]
    email: String,
    #[prost(int32, tag = "2")]
    sign_up_type: i32,
    #[prost(int64, tag = "3")]
    user_id: i64,
    #[prost(string, optional, tag = "4")]
    workos_id: Option<String>,
    #[prost(string, optional, tag = "5")]
    profile_picture_url: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct GetMeResponse {
    #[prost(string, tag = "1")]
    auth_id: String,
    #[prost(int32, tag = "2")]
    user_id: i32,
    #[prost(string, optional, tag = "3")]
    email: Option<String>,
    #[prost(string, optional, tag = "4")]
    first_name: Option<String>,
    #[prost(string, optional, tag = "5")]
    last_name: Option<String>,
    #[prost(string, optional, tag = "8")]
    created_at: Option<String>,
    #[prost(bool, optional, tag = "9")]
    is_enterprise_user: Option<bool>,
    #[prost(string, optional, tag = "11")]
    email_domain_type: Option<String>,
    #[prost(string, optional, tag = "12")]
    country: Option<String>,
    #[prost(string, optional, tag = "13")]
    profile_picture_url: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct GetUserProfileResponse {
    #[prost(bool, optional, tag = "4")]
    public_visibility_allowed: Option<bool>,
    #[prost(string, optional, tag = "5")]
    max_visibility: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct GetCurrentPeriodUsageResponse {
    #[prost(int64, tag = "1")]
    billing_cycle_start: i64,
    #[prost(int64, tag = "2")]
    billing_cycle_end: i64,
    #[prost(message, optional, tag = "3")]
    plan_usage: Option<PlanUsage>,
    #[prost(message, optional, tag = "4")]
    spend_limit_usage: Option<SpendLimitUsage>,
    #[prost(int32, optional, tag = "5")]
    display_threshold: Option<i32>,
    #[prost(bool, tag = "6")]
    enabled: bool,
    #[prost(string, tag = "7")]
    display_message: String,
    #[prost(string, optional, tag = "11")]
    auto_model_selected_display_message: Option<String>,
    #[prost(string, optional, tag = "12")]
    named_model_selected_display_message: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct PlanUsage {
    #[prost(int32, tag = "1")]
    total_spend: i32,
    #[prost(int32, tag = "2")]
    included_spend: i32,
    #[prost(int32, tag = "4")]
    remaining: i32,
    #[prost(int32, tag = "5")]
    limit: i32,
    #[prost(bool, optional, tag = "6")]
    remaining_bonus: Option<bool>,
    #[prost(string, optional, tag = "7")]
    bonus_tooltip: Option<String>,
    #[prost(int32, optional, tag = "8")]
    auto_spend: Option<i32>,
    #[prost(int32, optional, tag = "9")]
    api_spend: Option<i32>,
    #[prost(double, optional, tag = "12")]
    auto_percent_used: Option<f64>,
    #[prost(double, optional, tag = "13")]
    api_percent_used: Option<f64>,
    #[prost(double, optional, tag = "14")]
    total_percent_used: Option<f64>,
}

#[derive(Clone, PartialEq, Message)]
struct SpendLimitUsage {
    #[prost(string, tag = "8")]
    limit_type: String,
}

#[derive(Clone, PartialEq, Message)]
struct GetUsageLimitStatusAndActiveGrantsResponse {
    #[prost(message, optional, tag = "1")]
    usage_limit_policy_status: Option<UsageLimitPolicyStatus>,
}

#[derive(Clone, PartialEq, Message)]
struct UsageLimitPolicyStatus {
    #[prost(bool, tag = "1")]
    is_in_slow_pool: bool,
    #[prost(map = "string, string", tag = "5")]
    features: std::collections::HashMap<String, String>,
    #[prost(bool, tag = "6")]
    can_configure_spend_limit: bool,
    #[prost(bool, tag = "8")]
    has_pending_request: bool,
    #[prost(string, repeated, tag = "9")]
    allowed_model_ids: Vec<String>,
    #[prost(string, repeated, tag = "10")]
    allowed_model_tags: Vec<String>,
}

/// `GetUsageLimitStatusAndActiveGrantsResponse.model_recommendation_nudge`.
const MODEL_RECOMMENDATION_NUDGE_FIELD: u32 = 4;

#[derive(Clone, PartialEq, Message)]
struct ModelRecommendationNudge {
    #[prost(message, optional, tag = "4")]
    source_model: Option<agent::RequestedModel>,
}

#[derive(Clone, Copy, PartialEq, Message)]
struct Empty {}

pub async fn get_email(
    Extension(upstream): Extension<proxy::CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    local_or_forward(upstream, request, || {
        proto(GetEmailResponse {
            email: LOCAL_EMAIL.into(),
            sign_up_type: 3,
        })
    })
    .await
}

pub async fn get_user_meta(
    Extension(upstream): Extension<proxy::CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    local_or_forward(upstream, request, || {
        proto(GetUserMetaResponse {
            email: LOCAL_EMAIL.into(),
            sign_up_type: 3,
            user_id: 1,
            workos_id: None,
            profile_picture_url: None,
        })
    })
    .await
}

pub async fn get_me(
    Extension(upstream): Extension<proxy::CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    local_or_forward(upstream, request, || {
        proto(GetMeResponse {
            auth_id: LOCAL_AUTH_ID.into(),
            user_id: 1,
            email: Some(LOCAL_EMAIL.into()),
            first_name: Some("Cursor".into()),
            last_name: Some("Local".into()),
            created_at: Some(chrono::Utc::now().to_rfc3339()),
            is_enterprise_user: Some(false),
            email_domain_type: Some("personal".into()),
            country: Some("US".into()),
            profile_picture_url: None,
        })
    })
    .await
}

pub async fn get_teams(
    Extension(upstream): Extension<proxy::CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    local_or_forward(upstream, request, || proto(Empty {})).await
}

pub async fn get_user_profile(
    Extension(upstream): Extension<proxy::CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    local_or_forward(upstream, request, || {
        proto(GetUserProfileResponse {
            public_visibility_allowed: Some(true),
            max_visibility: Some("PUBLIC".into()),
        })
    })
    .await
}

pub async fn current_period_usage(
    Extension(upstream): Extension<proxy::CursorProxy>,
    Extension(free_entitlements): Extension<FreeEntitlementCache>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    local_or_confirmed_free_or_forward(upstream, &free_entitlements, request, || {
        let now = chrono::Utc::now();
        proto(GetCurrentPeriodUsageResponse {
            billing_cycle_start: (now - chrono::Duration::days(30)).timestamp_millis(),
            billing_cycle_end: (now + chrono::Duration::days(10 * 365)).timestamp_millis(),
            plan_usage: Some(PlanUsage {
                total_spend: 0,
                included_spend: LOCAL_ULTRA_PLAN_INCLUDED_CENTS,
                remaining: LOCAL_ULTRA_PLAN_INCLUDED_CENTS,
                limit: LOCAL_ULTRA_PLAN_INCLUDED_CENTS,
                remaining_bonus: Some(false),
                bonus_tooltip: Some("Ultra local account mock is active.".into()),
                auto_spend: Some(0),
                api_spend: Some(0),
                auto_percent_used: Some(0.0),
                api_percent_used: Some(0.0),
                total_percent_used: Some(0.0),
            }),
            spend_limit_usage: Some(SpendLimitUsage {
                limit_type: "user".into(),
            }),
            display_threshold: Some(99_999_999),
            enabled: true,
            display_message: "Ultra plan active".into(),
            auto_model_selected_display_message: Some("Ultra plan active".into()),
            named_model_selected_display_message: Some("Ultra plan active".into()),
        })
    })
    .await
}

pub async fn usage_limit_status(
    State(registry): State<TransportRegistry>,
    Extension(upstream): Extension<proxy::CursorProxy>,
    Extension(free_entitlements): Extension<FreeEntitlementCache>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    if local_app::request_uses_local_cursor_token(request.headers())
        || free_entitlements.is_confirmed_free(request.headers())
    {
        consume_body(request).await?;
        return proto(GetUsageLimitStatusAndActiveGrantsResponse {
            usage_limit_policy_status: Some(UsageLimitPolicyStatus {
                is_in_slow_pool: false,
                features: Default::default(),
                can_configure_spend_limit: true,
                has_pending_request: false,
                allowed_model_ids: Vec::new(),
                allowed_model_tags: Vec::new(),
            }),
        });
    }
    let response = proxy::forward_buffered(&upstream, request).await?;
    if !response.status.is_success() {
        return Ok(response.into_response());
    }
    let body = without_byok_model_nudge(registry.store(), &response.body).await?;
    Ok(response.with_body(body))
}

/// Removes Cursor's mid-conversation model nudge when it would switch a BYOK
/// conversation to another model.
///
/// After every successful foreground turn Cursor asks this endpoint for a
/// recommendation and applies a returned nudge silently, to the conversation
/// and to the global model selection ("Switched to Auto"). Cursor skips the
/// nudge only for its own API-key BYOK, so BYOK models here are switched to
/// Auto, whose smaller context window then forces Cursor to summarize the
/// conversation. Every other response field passes through byte for byte.
async fn without_byok_model_nudge(store: &Store, body: &Bytes) -> Result<Bytes> {
    let mut kept = BytesMut::with_capacity(body.len());
    let mut rest = &body[..];
    while !rest.is_empty() {
        let field = rest;
        let (tag, wire_type) = encoding::decode_key(&mut rest)?;
        let nudge = if tag == MODEL_RECOMMENDATION_NUDGE_FIELD
            && wire_type == encoding::WireType::LengthDelimited
        {
            let length = encoding::decode_varint(&mut rest)? as usize;
            if length > rest.len() {
                return Err(crate::Error::Protocol(
                    "truncated model recommendation nudge".into(),
                ));
            }
            let (nudge, after) = rest.split_at(length);
            rest = after;
            Some(ModelRecommendationNudge::decode(nudge)?)
        } else {
            encoding::skip_field(wire_type, tag, &mut rest, Default::default())?;
            None
        };
        let source_model = nudge
            .as_ref()
            .and_then(|nudge| nudge.source_model.as_ref())
            .map(|model| model.model_id.as_str());
        if let Some(model_id) = source_model {
            if is_local_model(store, model_id).await? {
                tracing::info!(
                    model_id,
                    "dropping Cursor model nudge away from a BYOK model"
                );
                continue;
            }
        }
        kept.extend_from_slice(&field[..field.len() - rest.len()]);
    }
    Ok(kept.freeze())
}

pub async fn stripe_profile(
    Extension(upstream): Extension<proxy::CursorProxy>,
    Extension(free_entitlements): Extension<FreeEntitlementCache>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    if local_app::request_uses_local_cursor_token(request.headers()) {
        return local_stripe_profile(request.headers().get(header::ORIGIN).cloned());
    }

    let request_headers = request.headers().clone();
    let origin = request.headers().get(header::ORIGIN).cloned();
    let upstream_response = match proxy::forward_buffered(&upstream, request).await {
        Ok(response) => response,
        Err(error) if free_entitlements.is_confirmed_free(&request_headers) => {
            tracing::warn!(%error, "using cached Free entitlement after Stripe upstream failure");
            return local_stripe_profile(origin);
        }
        Err(error) => return Err(error),
    };
    if !upstream_response.status.is_success() {
        if should_fallback_to_cached_free(
            upstream_response.status,
            &free_entitlements,
            &request_headers,
        ) {
            tracing::warn!(
                status = %upstream_response.status,
                "using cached Free entitlement after Stripe upstream failure"
            );
            return local_stripe_profile(origin);
        }
        return Ok(upstream_response.into_response());
    }

    let Some(membership_type) = membership_type(&upstream_response.body) else {
        return Ok(upstream_response.into_response());
    };
    let observed = free_entitlements.observe_membership(&request_headers, &membership_type);
    if observed && membership_type.eq_ignore_ascii_case("free") {
        return local_stripe_profile(origin);
    }

    Ok(upstream_response.into_response())
}

fn should_fallback_to_cached_free(
    status: axum::http::StatusCode,
    free_entitlements: &FreeEntitlementCache,
    headers: &axum::http::HeaderMap,
) -> bool {
    status.is_server_error() && free_entitlements.is_confirmed_free(headers)
}

fn membership_type(body: &[u8]) -> Option<String> {
    let profile: Value = serde_json::from_slice(body).ok()?;
    let membership_type = profile.get("membershipType")?.as_str()?.trim();
    (!membership_type.is_empty()).then(|| membership_type.to_owned())
}

fn local_stripe_profile(origin: Option<HeaderValue>) -> Result<Response<Body>> {
    let mut response = json(ultra_profile())?;
    if let Some(origin) = origin {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
            HeaderValue::from_static("true"),
        );
        response
            .headers_mut()
            .insert(header::VARY, HeaderValue::from_static("Origin"));
    }
    Ok(response)
}

async fn local_or_confirmed_free_or_forward(
    upstream: proxy::CursorProxy,
    free_entitlements: &FreeEntitlementCache,
    request: Request<Body>,
    local: impl FnOnce() -> Result<Response<Body>>,
) -> Result<Response<Body>> {
    if local_app::request_uses_local_cursor_token(request.headers())
        || free_entitlements.is_confirmed_free(request.headers())
    {
        consume_body(request).await?;
        return local();
    }
    proxy::forward(Extension(upstream), request).await
}

async fn local_or_forward(
    upstream: proxy::CursorProxy,
    request: Request<Body>,
    local: impl FnOnce() -> Result<Response<Body>>,
) -> Result<Response<Body>> {
    if local_app::request_uses_local_cursor_token(request.headers()) {
        consume_body(request).await?;
        return local();
    }
    proxy::forward(Extension(upstream), request).await
}

async fn consume_body(request: Request<Body>) -> Result<()> {
    to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|error| crate::Error::Protocol(format!("cannot read request body: {error}")))?;
    Ok(())
}

fn proto(message: impl Message) -> Result<Response<Body>> {
    response("application/proto", message.encode_to_vec())
}

fn json(value: Value) -> Result<Response<Body>> {
    response("application/json", serde_json::to_vec(&value)?)
}

fn response(content_type: &'static str, body: Vec<u8>) -> Result<Response<Body>> {
    let length = body.len();
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static(content_type),
    );
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        length
            .to_string()
            .parse()
            .expect("body length is always a valid header value"),
    );
    Ok(response)
}

fn ultra_profile() -> Value {
    serde_json::json!({
        "membershipType": "ultra",
        "individualMembershipType": "ultra",
        "subscriptionStatus": "active",
        "lastPaymentFailed": false,
        "pendingCancellationDate": null,
        "daysRemainingOnTrial": 0,
        "paymentId": LOCAL_AUTH_ID,
        "isTeamMember": false
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use axum::body::Bytes;
    use futures_util::stream;

    use super::*;

    #[tokio::test]
    async fn local_account_response_consumes_request_body_before_replying() {
        let polled = Arc::new(AtomicBool::new(false));
        let observed = polled.clone();
        let body = Body::from_stream(stream::once(async move {
            observed.store(true, Ordering::SeqCst);
            Ok::<_, std::convert::Infallible>(Bytes::from_static(b"request"))
        }));

        consume_body(Request::new(body)).await.unwrap();

        assert!(polled.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn drops_only_model_nudges_away_from_byok_models() {
        fn nudge_from(model_id: &str) -> Vec<u8> {
            let model = |model_id: &str| agent::RequestedModel {
                model_id: model_id.into(),
                ..Default::default()
            };
            let mut nudge = Vec::new();
            encoding::string::encode(1, &"nudge".to_owned(), &mut nudge);
            encoding::message::encode(4, &model(model_id), &mut nudge);
            encoding::message::encode(5, &model("default"), &mut nudge);
            let mut response = Vec::new();
            encoding::bytes::encode(MODEL_RECOMMENDATION_NUDGE_FIELD, &nudge, &mut response);
            response
        }
        // Usage status as Cursor sends it, including a field this crate does not model.
        let status = [0x0a, 0x08, 0x30, 0x01, 0x98, 0x01, 0x01, 0xd0, 0x01, 0x00];
        let directory = tempfile::tempdir().unwrap();
        let store = Store::connect(&format!(
            "sqlite://{}",
            directory.path().join("test.db").display()
        ))
        .await
        .unwrap();

        let byok = [
            &status[..],
            &nudge_from("plugin:dev.cursorbyok.examples.claude-auth/claude/claude-opus-5-5"),
        ]
        .concat();
        let filtered = without_byok_model_nudge(&store, &Bytes::from(byok))
            .await
            .unwrap();
        assert_eq!(&filtered[..], &status[..]);

        let official = [&status[..], &nudge_from("claude-4.5-opus-high")].concat();
        let filtered = without_byok_model_nudge(&store, &Bytes::from(official.clone()))
            .await
            .unwrap();
        assert_eq!(&filtered[..], &official[..]);
    }

    #[test]
    fn cached_free_fallback_accepts_server_failures_but_not_auth_failures() {
        let cache = FreeEntitlementCache::default();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer official-free-token"),
        );
        assert!(cache.observe_membership(&headers, "free"));

        assert!(should_fallback_to_cached_free(
            axum::http::StatusCode::BAD_GATEWAY,
            &cache,
            &headers
        ));
        assert!(!should_fallback_to_cached_free(
            axum::http::StatusCode::UNAUTHORIZED,
            &cache,
            &headers
        ));
    }

    #[test]
    fn reads_only_a_non_empty_membership_type() {
        assert_eq!(
            membership_type(br#"{"membershipType":"free"}"#).as_deref(),
            Some("free")
        );
        assert_eq!(membership_type(br#"{"membershipType":""}"#), None);
        assert_eq!(membership_type(br#"{"subscriptionStatus":"active"}"#), None);
        assert_eq!(membership_type(b"not-json"), None);
    }

    #[test]
    fn local_stripe_profile_allows_the_cursor_app_origin() {
        let origin = HeaderValue::from_static("vscode-file://vscode-app");
        let response = local_stripe_profile(Some(origin.clone())).unwrap();
        assert_eq!(
            response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&origin)
        );
        assert_eq!(
            response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_CREDENTIALS),
            Some(&HeaderValue::from_static("true"))
        );
        assert_eq!(
            response.headers().get(header::VARY),
            Some(&HeaderValue::from_static("Origin"))
        );
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("application/json"))
        );
    }
}

//! Chooses which plugin resource (account) a model call uses.
use super::state::ResourceRecord;

/// The ready resource whose included quota resets soonest, so quota that would otherwise expire
/// unused is spent first. Resources without an upcoming reset follow; ties keep creation order.
pub fn preferred(records: &[ResourceRecord], now_ms: i64) -> Option<&ResourceRecord> {
    records
        .iter()
        .filter(|record| record.state.is_ready(now_ms))
        .min_by_key(|record| {
            record
                .quota_reset_at_ms
                .filter(|reset_at_ms| *reset_at_ms > now_ms)
                .unwrap_or(i64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::state::ResourceState;

    fn record(id: &str, state: ResourceState, quota_reset_at_ms: Option<i64>) -> ResourceRecord {
        ResourceRecord {
            id: id.into(),
            key: id.into(),
            private_data: serde_json::Value::Null,
            state,
            quota_reset_at_ms,
            created_at_ms: 0,
            updated_at_ms: 0,
        }
    }

    fn preferred_id(records: &[ResourceRecord], now_ms: i64) -> Option<&str> {
        preferred(records, now_ms).map(|record| record.id.as_str())
    }

    #[test]
    fn spends_the_ready_resource_whose_quota_resets_first() {
        let records = [
            record("oct-29", ResourceState::Ready, Some(2_900)),
            record("unknown", ResourceState::Ready, None),
            record("oct-28", ResourceState::Ready, Some(2_800)),
        ];
        assert_eq!(preferred_id(&records, 100), Some("oct-28"));
    }

    #[test]
    fn skips_resources_that_are_not_ready() {
        let cooling = ResourceState::Cooling {
            retry_at_ms: Some(500),
            message: None,
        };
        let invalid = ResourceState::Invalid { message: None };
        let records = [
            record("cooling", cooling, Some(2_800)),
            record("invalid", invalid, Some(2_700)),
            record("oct-29", ResourceState::Ready, Some(2_900)),
        ];
        assert_eq!(preferred_id(&records, 100), Some("oct-29"));
        assert_eq!(
            preferred_id(&records, 600),
            Some("cooling"),
            "cooling expired"
        );
    }

    #[test]
    fn ranks_unknown_and_past_resets_last_in_creation_order() {
        let records = [
            record("past", ResourceState::Ready, Some(50)),
            record("unknown", ResourceState::Ready, None),
            record("oct-29", ResourceState::Ready, Some(2_900)),
        ];
        assert_eq!(preferred_id(&records, 100), Some("oct-29"));
        assert_eq!(preferred_id(&records[..2], 100), Some("past"));
        assert_eq!(preferred_id(&[], 100), None);
    }
}

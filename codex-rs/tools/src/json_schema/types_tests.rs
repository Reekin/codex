//! Verify serialization of the supported tool schema representation.

use super::JsonSchema;
use pretty_assertions::assert_eq;

#[test]
fn json_schema_serializes_encrypted_marker() {
    let schema = JsonSchema::string(Some("Secret value".to_string())).with_encrypted();

    assert_eq!(
        serde_json::to_value(schema).expect("serialize schema"),
        serde_json::json!({
            "type": "string",
            "description": "Secret value",
            "encrypted": true,
        })
    );
}

#[test]
fn json_schema_serializes_max_length() {
    let schema = JsonSchema::string(Some("Bounded value".to_string())).with_max_length(128);

    assert_eq!(
        serde_json::to_value(schema).expect("serialize schema"),
        serde_json::json!({
            "type": "string",
            "description": "Bounded value",
            "maxLength": 128,
        })
    );
}

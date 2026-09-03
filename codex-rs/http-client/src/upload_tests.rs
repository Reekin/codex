use super::*;

#[test]
fn low_rate_with_short_remaining_time_is_allowed() {
    let policy = UploadPolicy::default();
    let rate = 30 * 1024;
    let remaining = projected_duration(512 * 1024, rate);

    assert!(!upload_is_stalled(rate, remaining, policy));
}

#[test]
fn low_rate_with_long_remaining_time_is_stalled() {
    let policy = UploadPolicy::default();
    let rate = 30 * 1024;
    let remaining = projected_duration(8 * 1024 * 1024, rate);

    assert!(upload_is_stalled(rate, remaining, policy));
}

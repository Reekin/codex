use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

fn request(input: serde_json::Value) -> ResponsesApiRequest {
    ResponsesApiRequest {
        model: "explicit-model".into(),
        instructions: "keep instructions".into(),
        input: serde_json::from_value(input).unwrap(),
        tools: Some(
            Arc::<serde_json::value::RawValue>::from(
                serde_json::value::RawValue::from_string(
                    r#"[{"type":"function","name":"look","parameters":{"type":"object"}}]"#.into(),
                )
                .unwrap(),
            )
            .into(),
        ),
        tool_choice: "auto".into(),
        parallel_tool_calls: false,
        reasoning: None,
        store: false,
        stream: true,
        stream_options: None,
        include: vec![],
        service_tier: None,
        prompt_cache_key: Some("explicit-cache-key".into()),
        text: None,
        client_metadata: None,
        access_programs: None,
    }
}

#[test]
fn request_input_budget_preserves_text_tools_and_latest_images() {
    let original = request(json!([
        {"type":"message","role":"user","content":[
            {"type":"input_text","text":"keep user text"},
            {"type":"input_image","image_url":"a".repeat(600)}]},
        {"type":"function_call","call_id":"call-1","name":"look","arguments":"{}"},
        {"type":"function_call_output","call_id":"call-1","output":[
            {"type":"input_text","text":"keep tool text"},
            {"type":"input_image","image_url":"b".repeat(600)}]},
        {"type":"custom_tool_call","call_id":"call-2","name":"look","input":""},
        {"type":"custom_tool_call_output","call_id":"call-2","output":[
            {"type":"input_image","image_url":"c".repeat(600)}]}
    ]));
    let budget = json_bytes(&original).unwrap() as u64 - 700;
    let limits = Some(ModelInputLimits {
        max_request_bytes: std::num::NonZeroU64::new(budget),
        ..Default::default()
    });
    let mut projected = original.clone();
    prepare(
        &mut projected,
        limits,
        &OmittedImages::default(),
        json_bytes,
    )
    .unwrap();
    let mut expected = serde_json::to_value(&original).unwrap();
    expected["input"][0]["content"][1] = json!({"type":"input_text","text":OMITTED_IMAGE});
    expected["input"][2]["output"][1] = json!({"type":"input_text","text":OMITTED_IMAGE});
    assert_eq!(serde_json::to_value(&projected).unwrap(), expected);
    assert!(json_bytes(&projected).unwrap() as u64 <= budget);
    let mut retry = original.clone();
    prepare(&mut retry, limits, &OmittedImages::default(), json_bytes).unwrap();
    assert_eq!(retry, projected);
    let mut resumed: ResponsesApiRequest = original.clone();
    resumed.input = serde_json::from_slice(&serde_json::to_vec(&original.input).unwrap()).unwrap();
    resumed.input.push(
        serde_json::from_value(
            json!({"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}),
        )
        .unwrap(),
    );
    prepare(&mut resumed, limits, &OmittedImages::default(), json_bytes).unwrap();
    assert_eq!(resumed.input[..original.input.len()], projected.input);
    let mut unconfigured = original.clone();
    prepare(
        &mut unconfigured,
        None,
        &OmittedImages::default(),
        json_bytes,
    )
    .unwrap();
    assert_eq!(unconfigured, original);
}

#[test]
fn request_input_non_image_budget_counts_instructions_and_tools() {
    let mut request = request(json!([]));
    let exact = json_bytes(&request).unwrap() as u64;
    let limits = |bytes| {
        Some(ModelInputLimits {
            max_request_bytes: std::num::NonZeroU64::new(bytes),
            ..Default::default()
        })
    };
    prepare(
        &mut request,
        limits(exact),
        &OmittedImages::default(),
        json_bytes,
    )
    .unwrap();
    let error = prepare(
        &mut request,
        limits(exact - 1),
        &OmittedImages::default(),
        json_bytes,
    )
    .unwrap_err();
    assert!(error.to_string().contains("non-image content requires"));
}

#[test]
fn request_input_dimensions_cache_and_model_switch_are_deterministic() {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(100, 50)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let original = codex_utils_image::data_url_from_bytes("image/png", bytes.get_ref());
    let small = prepared_image(&original, 40).unwrap();
    let again = prepared_image(&original, 40).unwrap();
    assert!(Arc::ptr_eq(&small, &again));
    let decoded = load_data_url_for_prompt_uncached(&small, PromptImageMode::Original).unwrap();
    assert_eq!((decoded.width, decoded.height), (40, 20));
    assert_eq!(&*prepared_image(&original, 100).unwrap(), original);
    assert_eq!(prepared_image(&original, 40).unwrap(), small);
}

#[test]
fn request_input_limits_reject_zero_and_unknown_fields() {
    for value in [
        json!({"max_image_dimension":0}),
        json!({"max_request_bytes":0}),
        json!({"max_image_pixels":10}),
    ] {
        assert!(serde_json::from_value::<ModelInputLimits>(value).is_err());
    }
    for value in [
        json!({"max_image_dimension":42}),
        json!({"max_request_bytes":1024}),
    ] {
        let limits: ModelInputLimits = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(limits).unwrap(), value);
    }
}

#[test]
fn request_input_large_history_budget_is_repeatable() {
    let mut state = 13u32;
    let pixels = (0..512 * 512 * 3)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        })
        .collect();
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(512, 512, pixels).unwrap())
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let url = codex_utils_image::data_url_from_bytes("image/png", bytes.get_ref());
    let mut original = request(json!([]));
    for index in 0..40_000_000usize.div_ceil(url.len()) {
        original.input.push(
            serde_json::from_value(json!({"type":"message","role":"user","content":[
                {"type":"input_text","text":format!("image {index}")},
                {"type":"input_image","image_url":url}
            ]}))
            .unwrap(),
        );
    }
    let limits = Some(
        serde_json::from_value(json!({"max_image_dimension":2000,"max_request_bytes":30000000}))
            .unwrap(),
    );
    let before = json_bytes(&original).unwrap();
    let mut prepared = original.clone();
    prepare(&mut prepared, limits, &OmittedImages::default(), json_bytes).unwrap();
    let after = json_bytes(&prepared).unwrap();
    eprintln!("request_input_large_history: before={before} after={after} budget=30000000");
    assert!(before >= 40_000_000);
    assert!(after <= 30_000_000);
    let mut repeated = original;
    prepare(&mut repeated, limits, &OmittedImages::default(), json_bytes).unwrap();
    assert_eq!(prepared, repeated);
}

#[test]
fn request_input_budget_omits_in_steps_and_keeps_earlier_omissions() {
    let image = |index: usize| {
        json!({"type":"message","role":"user","content":[
            {"type":"input_image","image_url":format!("{index:04}{}", "x".repeat(996))}
        ]})
    };
    let history = |count: usize| request(json!((0..count).map(image).collect::<Vec<_>>()));
    let limit = json_bytes(&history(10)).unwrap() as u64;
    let limits = Some(ModelInputLimits {
        max_request_bytes: std::num::NonZeroU64::new(limit),
        ..Default::default()
    });
    let omitted = OmittedImages::default();
    let mut previous: Option<ResponsesApiRequest> = None;
    let mut prefix_changes = 0;
    for count in 1..=30 {
        let mut prepared = history(count);
        prepare(&mut prepared, limits, &omitted, json_bytes).unwrap();
        assert!(json_bytes(&prepared).unwrap() as u64 <= limit);
        let newest = serde_json::to_value(prepared.input.last().unwrap()).unwrap();
        assert_eq!(newest, image(count - 1), "the newest image is kept");
        if let Some(previous) = &previous
            && !prepared.input.starts_with(&previous.input)
        {
            prefix_changes += 1;
        }
        previous = Some(prepared);
    }
    // Twenty images past the budget change the prefix a few times, not on every image.
    assert!((3..=6).contains(&prefix_changes), "{prefix_changes}");
}

#[test]
fn request_input_opaque_images_become_jpeg_and_transparent_ones_stay_png() {
    let mut state = 29u32;
    let mut noise = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    };
    let opaque = image::RgbImage::from_fn(256, 256, |_, _| image::Rgb([noise(), noise(), noise()]));
    let transparent = image::RgbaImage::from_fn(256, 256, |_, _| {
        image::Rgba([noise(), noise(), noise(), 128])
    });
    let url = |image: image::DynamicImage| {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        codex_utils_image::data_url_from_bytes("image/png", bytes.get_ref())
    };
    let opaque_url = url(image::DynamicImage::ImageRgb8(opaque));
    let prepared = prepared_image(&opaque_url, 2000).unwrap();
    assert!(prepared.starts_with("data:image/jpeg;base64,"));
    assert!(prepared.len() < opaque_url.len());
    let decoded = load_data_url_for_prompt_uncached(&prepared, PromptImageMode::Original).unwrap();
    assert_eq!((decoded.width, decoded.height), (256, 256));
    let transparent_url = url(image::DynamicImage::ImageRgba8(transparent));
    assert_eq!(
        &*prepared_image(&transparent_url, 2000).unwrap(),
        transparent_url
    );
}
